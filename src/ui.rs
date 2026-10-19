//! Provides `UiShape` component for rendering SDF shapes in Bevy's UI

use bevy::{
    ecs::entity::EntityHashMap,
    ecs::system::{
        SystemParamItem,
        lifetimeless::{Read, SRes},
    },
    math::{Affine2, Rect},
    prelude::*,
    render::{
        Extract, ExtractSchedule, GpuResourceAppExt, MainWorld, Render, RenderApp, RenderSystems,
        globals::{GlobalsBuffer, GlobalsUniform},
        render_phase::{
            AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctions, PhaseItem,
            PhaseItemExtraIndex, RenderCommand, RenderCommandResult, SetItemPipeline,
            TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BlendComponent, BlendFactor, BlendOperation, BlendState, BufferUsages,
            CachedPipelineState, CachedRenderPipelineId, ColorTargetState, ColorWrites,
            FragmentState, FrontFace, MultisampleState, PipelineCache, PolygonMode, PrimitiveState,
            PrimitiveTopology, RawBufferVec, RenderPipelineDescriptor, ShaderStages,
            SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat, VertexAttribute,
            VertexFormat, VertexState, VertexStepMode, binding_types::uniform_buffer,
        },
        renderer::{RenderDevice, RenderQueue},
        sync_world::{MainEntity, TemporaryRenderEntity},
        view::{ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
    ui::{
        CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiTargetCamera, Node,
        UiGlobalTransform,
    },
    ui_render::{TransparentUi, UiCameraMap, UiCameraView, stack_z_offsets},
};
use bytemuck::{Pod, Zeroable};

use crate::{
    BlendMode, FloatOrd, GeneratedShaders, PendingShaders, SIMPLE_FILL_HANDLE, VertexBufferLayout,
    shader_loading::VERTEX_SHADER_HANDLE,
};

/// Component for rendering shapes in UI.
///
/// This component requires `Node` and renders an SDF-based shape within the UI node bounds.
#[derive(Component, Reflect, Debug, Clone)]
#[require(Node)]
#[reflect(Component)]
pub struct UiShape {
    /// The color of the shape
    pub color: Color,

    /// Shader containing a wgsl function for a signed distance field
    ///
    /// The shader needs to have the signature `fn sdf(input: smud::SdfInput) -> f32`.
    pub sdf: Handle<Shader>,

    /// Shader containing a wgsl function for the fill of the shape
    ///
    /// The shader needs to have the signature `fn fill(input: smud::FillInput) -> vec4<f32>`.
    pub fill: Handle<Shader>,

    /// Parameters to pass to shapes, for things such as width of a box
    pub params: Vec4,

    /// Blend mode for the shape
    pub blend_mode: BlendMode,

    /// Extra padding to add to the bounds when rendering the shape
    pub extra_bounds: f32,

    /// Second parameter vector for fills (`FillInput.aux`); zero for the
    /// world-space `SmudShape` path
    pub aux: Vec4,

    /// Second colour for fills (`FillInput.aux_color`, linear); zero for the
    /// world-space `SmudShape` path
    pub aux_color: Color,
}

impl Default for UiShape {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            sdf: Handle::default(),
            fill: SIMPLE_FILL_HANDLE.clone(),
            params: Vec4::ZERO,
            blend_mode: BlendMode::default(),
            extra_bounds: 0.0,
            aux: Vec4::ZERO,
            aux_color: Color::NONE,
        }
    }
}

impl UiShape {
    /// Set the blend mode for this shape (builder pattern)
    pub fn with_blend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.blend_mode = blend_mode;
        self
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UiShapeVertex {
    position: [f32; 3],
    color: [f32; 4],
    params: [f32; 4],
    rotation: [f32; 2],
    scale: f32,
    bounds: [f32; 4],
    /// Clip rect in target pixels (`min.xy`, `max.xy`)
    clip: [f32; 4],
    /// Second fill parameters (`FillInput.aux`)
    aux: [f32; 4],
    /// Second fill colour (`FillInput.aux_color`)
    aux_color: [f32; 4],
}

#[derive(Resource)]
struct UiShapeMeta {
    vertices: RawBufferVec<UiShapeVertex>,
    view_bind_group: Option<BindGroup>,
}

impl Default for UiShapeMeta {
    fn default() -> Self {
        Self {
            vertices: RawBufferVec::new(BufferUsages::VERTEX),
            view_bind_group: None,
        }
    }
}

struct ExtractedUiShape {
    main_entity: MainEntity,
    render_entity: Entity,
    stack_index: u32,
    transform: Affine2,
    /// Node bounds in local space
    rect: Rect,
    extra_bounds: f32,
    color: Color,
    params: Vec4,
    aux: Vec4,
    aux_color: Color,
    shader: Handle<Shader>,
    blend_mode: BlendMode,
    /// `CalculatedClip` of the node (none = unclipped)
    clip: Option<Rect>,
    /// Render entity of the node's target camera.
    extracted_camera_entity: Entity,
}

#[derive(Resource, Default)]
struct ExtractedUiShapes {
    nodes: Vec<ExtractedUiShape>,
}

// Generates the combined sdf+fill shaders for added and modified UI shapes.
// Shapes whose shader assets have not loaded yet are remembered in
// `PendingShaders` (shared with the world-space path) and retried until their
// assets become available.
fn generate_shaders(
    mut main_world: ResMut<MainWorld>,
    mut generated_shaders: ResMut<GeneratedShaders>,
    mut pending_shaders: ResMut<PendingShaders>,
) {
    main_world.resource_scope(|world, mut shaders: Mut<Assets<Shader>>| {
        // Retry pairs that were waiting for their shader assets to load
        pending_shaders.0.retain(|(sdf, fill)| {
            generated_shaders
                .try_generate(sdf, fill, &mut shaders)
                .is_none()
        });

        // Only scan shapes that were added or modified since the last frame
        let mut ui_shapes = world.query_filtered::<&UiShape, Changed<UiShape>>();
        for ui_shape in ui_shapes.iter(world) {
            if generated_shaders
                .try_generate(&ui_shape.sdf, &ui_shape.fill, &mut shaders)
                .is_none()
            {
                let pair = (ui_shape.sdf.clone(), ui_shape.fill.clone());
                if !pending_shaders.0.contains(&pair) {
                    pending_shaders.0.push(pair);
                }
            }
        }
    });
}

/// Extract UiShape components to render world
#[allow(clippy::type_complexity)]
fn extract_ui_shapes(
    mut commands: Commands,
    mut extracted_nodes: ResMut<ExtractedUiShapes>,
    generated_shaders: Res<GeneratedShaders>,
    ui_shapes: Extract<
        Query<(
            Entity,
            &UiShape,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    extracted_nodes.nodes.clear();
    let mut camera_mapper = camera_map.get_mapper();

    for (entity, ui_shape, computed_node, stack_index, transform, visibility, clip, target) in
        ui_shapes.iter()
    {
        // hidden nodes (and zero-sized ones) are not drawn,
        // matching bevy_ui's own extraction.
        if !visibility.get() || computed_node.is_empty() {
            continue;
        }

        // draw only into the node's own UI camera (like
        // bevy_ui), not into every view with a UI phase.
        let Some(extracted_camera_entity) = camera_mapper.map(target) else {
            continue;
        };

        let Some(clip) = clip.map_or(Some(None), clip_to_rect) else {
            // Fully clipped by an ancestor
            continue;
        };

        let Some(shader) = generated_shaders
            .0
            .get(&(ui_shape.sdf.id(), ui_shape.fill.id()))
            .cloned()
        else {
            // Shader not yet generated - skip this node for now
            continue;
        };

        let render_entity = commands.spawn(TemporaryRenderEntity::default()).id();

        extracted_nodes.nodes.push(ExtractedUiShape {
            main_entity: entity.into(),
            render_entity,
            stack_index: stack_index.0,
            transform: transform.into(),
            rect: Rect {
                min: Vec2::ZERO,
                max: computed_node.size,
            },
            extra_bounds: ui_shape.extra_bounds,
            color: ui_shape.color,
            params: ui_shape.params,
            aux: ui_shape.aux,
            aux_color: ui_shape.aux_color,
            shader,
            blend_mode: ui_shape.blend_mode,
            clip,
            extracted_camera_entity,
        });
    }
}

/// Clip rect extent used for unclipped edges
const UNCLIPPED: f32 = 1.0e9;

/// Reduces a node's `CalculatedClip` to the single axis-aligned rect (in
/// target pixels) the shader clips against.
///
/// Returns `None` if the node is fully clipped, `Some(None)` if it is not
/// clipped at all. Exact for axis-aligned clip ancestors; for rotated ones the
/// clip region is approximated by its bounding box.
fn clip_to_rect(clip: &CalculatedClip) -> Option<Option<Rect>> {
    let rects = clip.rects()?;
    let mut result: Option<Rect> = None;
    for clip_rect in rects {
        let clip_local_to_world = clip_rect.world_to_clip_local.inverse();
        // Clipping along a single axis leaves the other edges infinite, which
        // would turn into NaN when transformed
        let min = clip_rect.rect.min.max(Vec2::splat(-UNCLIPPED));
        let max = clip_rect.rect.max.min(Vec2::splat(UNCLIPPED));
        let corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)]
            .map(|corner| clip_local_to_world.transform_point2(corner));
        let world_rect = corners.iter().fold(
            Rect::from_corners(corners[0], corners[0]),
            |rect, &corner| rect.union_point(corner),
        );
        let intersection = result.map_or(world_rect, |result| result.intersect(world_rect));
        if intersection.is_empty() {
            return None;
        }
        result = Some(intersection);
    }
    Some(result)
}

/// Pipeline key for specializing UI rendering based on shaders
#[derive(Clone, Hash, PartialEq, Eq)]
struct UiShapePipelineKey {
    shader: Handle<Shader>,
    blend_mode: BlendMode,
    /// The view's render target format: `Rgba16Float` on an HDR camera.
    target_format: TextureFormat,
}

/// Pipeline for rendering shapes in UI.
#[derive(Resource)]
struct UiShapePipeline {
    view_layout: BindGroupLayoutDescriptor,
}

impl Default for UiShapePipeline {
    fn default() -> Self {
        let entries = BindGroupLayoutEntries::with_indices(
            ShaderStages::VERTEX_FRAGMENT,
            (
                (0, uniform_buffer::<ViewUniform>(true)),
                (
                    1,
                    uniform_buffer::<GlobalsUniform>(false).visibility(ShaderStages::FRAGMENT),
                ),
            ),
        );

        let view_layout = BindGroupLayoutDescriptor::new("ui_shape_view_layout", &entries);

        Self { view_layout }
    }
}

impl SpecializedRenderPipeline for UiShapePipeline {
    type Key = UiShapePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // Get the generated shader for this sdf+fill combination
        let shader = key.shader;

        RenderPipelineDescriptor {
            label: Some("ui_shape_pipeline".into()),
            layout: vec![self.view_layout.clone()],
            immediate_size: 0,
            vertex: VertexState {
                shader: VERTEX_SHADER_HANDLE,
                shader_defs: vec!["Y_DOWN".into(), "UI_CLIP".into()],
                constants: Vec::new(),
                entry_point: Some("vertex".into()),
                buffers: vec![VertexBufferLayout {
                    array_stride: std::mem::size_of::<UiShapeVertex>() as u64,
                    step_mode: VertexStepMode::Instance, // One instance per UI node
                    attributes: vec![
                        // position
                        VertexAttribute {
                            format: VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        // color
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 12,
                            shader_location: 1,
                        },
                        // params
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 28,
                            shader_location: 2,
                        },
                        // rotation
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 44,
                            shader_location: 3,
                        },
                        // scale
                        VertexAttribute {
                            format: VertexFormat::Float32,
                            offset: 52,
                            shader_location: 4,
                        },
                        // bounds
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 56,
                            shader_location: 5,
                        },
                        // clip rect
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 72,
                            shader_location: 6,
                        },
                        // aux
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 88,
                            shader_location: 7,
                        },
                        // aux_color
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 104,
                            shader_location: 8,
                        },
                    ],
                }],
            },
            fragment: Some(FragmentState {
                shader,
                shader_defs: vec!["UI_CLIP".into()],
                constants: Vec::new(),
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(match key.blend_mode {
                        BlendMode::Alpha => BlendState::ALPHA_BLENDING,
                        BlendMode::Additive => BlendState {
                            color: BlendComponent {
                                src_factor: BlendFactor::SrcAlpha,
                                dst_factor: BlendFactor::One,
                                operation: BlendOperation::Add,
                            },
                            alpha: BlendComponent {
                                src_factor: BlendFactor::One,
                                dst_factor: BlendFactor::One,
                                operation: BlendOperation::Add,
                            },
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            zero_initialize_workgroup_memory: false,
        }
    }
}

/// Prepares the vertex buffers for UI shapes and batches them.
///
/// Iterates each view's sorted `TransparentUi` phase, writes one instance per
/// item in draw order, and merges consecutive items that share a pipeline into
/// a single batch (and thereby a single draw call), like bevy_ui's
/// `prepare_uinodes` does for its own nodes.
#[allow(clippy::type_complexity)]
fn prepare_ui_shapes(
    mut ui_shape_meta: ResMut<UiShapeMeta>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted_nodes: Res<ExtractedUiShapes>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    view_uniforms: Res<ViewUniforms>,
    globals_buffer: Res<GlobalsBuffer>,
    pipeline: Res<UiShapePipeline>,
    pipeline_cache: Res<PipelineCache>,
) {
    // Create view bind group
    if let (Some(view_binding), Some(globals)) = (
        view_uniforms.uniforms.binding(),
        globals_buffer.buffer.binding(),
    ) {
        let view_layout = pipeline_cache.get_bind_group_layout(&pipeline.view_layout);
        ui_shape_meta.view_bind_group = Some(render_device.create_bind_group(
            "ui_shape_view_bind_group",
            &view_layout,
            &BindGroupEntries::with_indices(((0, view_binding), (1, globals))),
        ));
    }

    ui_shape_meta.vertices.clear();

    let mut instance_index: u32 = 0;

    // Phase items no longer carry an index into the extracted nodes, so look
    // nodes up by their render entity instead
    let node_indices: EntityHashMap<usize> = extracted_nodes
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.render_entity, index))
        .collect();

    for (_, transparent_phase) in transparent_render_phases.iter_mut() {
        // Batches must never span items with different pipelines, since the
        // phase renderer only draws the leader item of a batch range
        let mut batch_pipeline = CachedRenderPipelineId::INVALID;
        let mut batch_item_index = 0;
        let mut batch_first_instance = 0;

        for item_index in 0..transparent_phase.items.len() {
            let item = &transparent_phase.items[item_index];
            let item_pipeline = item.cached_pipeline();
            let item_entity = item.entity();

            let Some(node) = node_indices
                .get(&item_entity)
                .map(|&index| &extracted_nodes.nodes[index])
            else {
                // Not one of our shape items: start a new batch to respect draw order
                batch_pipeline = CachedRenderPipelineId::INVALID;
                continue;
            };

            // Extract transform components from Affine2
            let position = node.transform.translation;

            // Extract rotation and scale from the matrix
            let x_axis = node.transform.matrix2.x_axis;
            let y_axis = node.transform.matrix2.y_axis;

            // Scale is the length of the axis vectors
            let scale_x = x_axis.length();
            let scale_y = y_axis.length();
            let scale = (scale_x + scale_y) / 2.0; // Average scale (assuming uniform)

            // Rotation is the direction of the x-axis (normalized)
            let rotation = if scale_x > 0.0 {
                let normalized = x_axis / scale_x;
                [normalized.x, normalized.y] // [cos, sin]
            } else {
                [1.0, 0.0] // No rotation
            };

            let bounds = node.rect.size() / 2.0;

            ui_shape_meta.vertices.push(UiShapeVertex {
                position: [position.x, position.y, 0.0],
                color: node.color.to_linear().to_f32_array(),
                params: node.params.to_array(),
                rotation,
                scale,
                bounds: [bounds.x, bounds.y, node.extra_bounds, node.extra_bounds],
                clip: node
                    .clip
                    .map_or([-UNCLIPPED, -UNCLIPPED, UNCLIPPED, UNCLIPPED], |clip| {
                        [clip.min.x, clip.min.y, clip.max.x, clip.max.y]
                    }),
                aux: node.aux.to_array(),
                aux_color: node.aux_color.to_linear().to_f32_array(),
            });

            if item_pipeline != batch_pipeline {
                batch_pipeline = item_pipeline;
                batch_item_index = item_index;
                batch_first_instance = instance_index;
            }

            // The leader item draws every instance in the batch range, while
            // the follower items are skipped by the phase renderer. The range
            // is stored in instance coordinates: since every item corresponds
            // to exactly one instance, its length also matches the number of
            // items the phase renderer has to skip.
            *transparent_phase.items[batch_item_index].batch_range_mut() =
                batch_first_instance..(instance_index + 1);

            instance_index += 1;
        }
    }

    ui_shape_meta
        .vertices
        .write_buffer(&render_device, &render_queue);
}

fn queue_ui_shapes(
    draw_functions: Res<DrawFunctions<TransparentUi>>,
    pipeline: Res<UiShapePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<UiShapePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<&UiCameraView, With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
    extracted_nodes: Res<ExtractedUiShapes>,
) {
    let draw_function = draw_functions.read().id::<DrawUiShapes>();

    // The camera -> (view, phase) resolution is cached across consecutive
    // nodes, since all shapes usually target the same camera
    // (like bevy_ui's `queue_uinodes`).
    let mut current_camera_entity = Entity::PLACEHOLDER;
    let mut current_view: Option<&ExtractedView> = None;
    let mut current_phase = None;

    // Each node goes into the UI phase of its own target camera
    // (bevy_ui's `queue_uinodes` does the same). The pipeline is specialized
    // on that view's target format: an HDR camera renders UI to
    // `Rgba16Float`, and a pipeline built for `Rgba8UnormSrgb` fails
    // validation there.
    for node in &extracted_nodes.nodes {
        if node.extracted_camera_entity != current_camera_entity {
            current_camera_entity = node.extracted_camera_entity;
            current_view = render_views
                .get(node.extracted_camera_entity)
                .ok()
                .and_then(|ui_view| camera_views.get(ui_view.0).ok());
            current_phase = current_view
                .and_then(|view| transparent_render_phases.get_mut(&view.retained_view_entity));
        }

        let (Some(view), Some(transparent_phase)) = (current_view, current_phase.as_deref_mut())
        else {
            continue;
        };

        // Create pipeline key for this shader combination
        let key = UiShapePipelineKey {
            shader: node.shader.clone(),
            blend_mode: node.blend_mode,
            target_format: view.target_format,
        };

        // Specialize the pipeline for this shader combination
        let pipeline_id = pipelines.specialize(&pipeline_cache, &pipeline, key);

        // Check if pipeline is ready - if not, skip this node
        if !matches!(
            pipeline_cache.get_render_pipeline_state(pipeline_id),
            CachedPipelineState::Ok(_)
        ) {
            continue;
        }

        // Use stack_index with an offset to control z-ordering
        // We use a value slightly after MATERIAL (0.05) so UiShapes render in
        // proper layer order. The offset must be unique per node type to avoid
        // z-fighting with bevy_ui's own stack offsets (TEXT_SELECTION is 0.055,
        // TEXT is 0.06), so we use 0.0025 (0.0525 in total).
        let sort_key = FloatOrd(node.stack_index as f32 + stack_z_offsets::MATERIAL + 0.0025);

        // batch_range is calculated in prepare_ui_shapes
        transparent_phase.add_transient(TransparentUi {
            entity: (node.render_entity, node.main_entity),
            draw_function,
            pipeline: pipeline_id,
            sort_key,
            batch_range: 0..0,
            extra_index: PhaseItemExtraIndex::None,
            indexed: false,
        });
    }
}

type DrawUiShapes = (
    SetItemPipeline,
    SetUiShapeViewBindGroup<0>,
    DrawUiShapeBatch,
);

struct SetUiShapeViewBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetUiShapeViewBindGroup<I> {
    type Param = SRes<UiShapeMeta>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        ui_shape_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(view_bind_group) = ui_shape_meta.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

struct DrawUiShapeBatch;

impl RenderCommand<TransparentUi> for DrawUiShapeBatch {
    type Param = SRes<UiShapeMeta>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &TransparentUi,
        _view: (),
        _entity: Option<()>,
        ui_shape_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let ui_shape_meta = ui_shape_meta.into_inner();
        let Some(vertices) = ui_shape_meta.vertices.buffer() else {
            return RenderCommandResult::Failure("no vertex buffer");
        };

        pass.set_vertex_buffer(0, vertices.slice(..));
        // Draw every shape in this batch with a single call. The batch range
        // holds a range of instances in the shared vertex buffer (see
        // `prepare_ui_shapes`).
        pass.draw(0..4, item.batch_range.clone());
        RenderCommandResult::Success
    }
}

/// Plugin for rendering shapes in bevy_ui
pub(crate) struct UiShapePlugin;

impl Plugin for UiShapePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<UiShape>();
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawUiShapes>()
                .init_gpu_resource::<UiShapeMeta>()
                .init_resource::<UiShapePipeline>()
                .init_gpu_resource::<SpecializedRenderPipelines<UiShapePipeline>>()
                .init_resource::<ExtractedUiShapes>()
                .add_systems(
                    ExtractSchedule,
                    (generate_shaders, extract_ui_shapes.after(generate_shaders)),
                )
                .add_systems(
                    Render,
                    (
                        queue_ui_shapes.in_set(RenderSystems::Queue),
                        // batching needs the phases to be sorted first
                        prepare_ui_shapes.in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}
