use std::borrow::Cow;

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    prelude::*,
};

const SMUD_SHADER_HANDLE: Handle<Shader> = uuid_handle!("eca0ed31-c377-41be-94a8-55154ecb0810");

const VIEW_BINDINGS_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("f973016d-a9cc-469a-afb6-64ad829cd838");

pub const VERTEX_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("27b9d87f-6a69-49ee-a2e8-c0bc08ee4f61");

/// The default fill used by `SmudShape`
pub const DEFAULT_FILL_HANDLE: Handle<Shader> =
    uuid_handle!("30981e86-7600-4089-b4e7-992601dc96b4");

/// Simple single-colored filled fill
pub const SIMPLE_FILL_HANDLE: Handle<Shader> = uuid_handle!("cef2d2c2-1a68-4418-a815-5a8ac361f140");

/// Creates an internal WESL shader importable under `module_path`
/// (e.g. `smud::shapes::circle`).
pub(crate) fn wesl_module(source: impl Into<Cow<'static, str>>, module_path: &str) -> Shader {
    // bevy derives the module path of `embedded://` shaders from their path
    Shader::from_wesl(
        source,
        format!("embedded://{}.wesl", module_path.replace("::", "/")),
    )
}

pub struct ShaderLoadingPlugin;

impl Plugin for ShaderLoadingPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SMUD_SHADER_HANDLE,
            "../assets/smud.wesl",
            |source, _| wesl_module(source, "smud")
        );

        load_internal_asset!(
            app,
            VIEW_BINDINGS_SHADER_HANDLE,
            "../assets/view_bindings.wesl",
            |source, _| wesl_module(source, "bevy_smud::view_bindings")
        );

        load_internal_asset!(
            app,
            VERTEX_SHADER_HANDLE,
            "../assets/vertex.wesl",
            |source, _| wesl_module(source, "smud::vertex")
        );

        load_internal_asset!(
            app,
            DEFAULT_FILL_HANDLE,
            "../assets/fills/cubic_falloff.wesl",
            |source, _| wesl_module(source, "smud::default_fill")
        );

        load_internal_asset!(
            app,
            SIMPLE_FILL_HANDLE,
            "../assets/fills/simple.wesl",
            |source, _| wesl_module(source, "smud::simple_fill")
        );
    }
}
