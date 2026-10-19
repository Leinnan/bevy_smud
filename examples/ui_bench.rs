//! Benchmark for rendering many `UiShape` nodes in the UI render phase.
//!
//! Controls:
//! - `SMUD_UI_BENCH_COUNT`: number of shapes (default: 20000)
//! - `SMUD_UI_BENCH_INTERLEAVE=1`: cycle through multiple sdf shaders and blend modes,
//!   which forces pipeline changes between neighboring nodes

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
};
use bevy_asset_loader::prelude::*;
use bevy_smud::prelude::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            LogDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            SmudPlugin,
        ))
        .init_state::<GameState>()
        .add_loading_state(
            LoadingState::new(GameState::Loading)
                .continue_to_state(GameState::Running)
                .load_collection::<AssetHandles>(),
        )
        .add_systems(OnEnter(GameState::Running), setup)
        .run();
}

#[derive(Clone, Eq, PartialEq, Debug, Hash, States, Default)]
enum GameState {
    #[default]
    Loading,
    Running,
}

#[derive(Resource, AssetCollection)]
struct AssetHandles {
    #[asset(path = "shapes/circle.wesl")]
    circle: Handle<Shader>,
    #[asset(path = "shapes/rectangle.wesl")]
    rectangle: Handle<Shader>,
    #[asset(path = "fills/simple.wesl")]
    simple_fill: Handle<Shader>,
}

fn setup(mut commands: Commands, assets: Res<AssetHandles>) {
    let count: usize = std::env::var("SMUD_UI_BENCH_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let interleave = std::env::var("SMUD_UI_BENCH_INTERLEAVE").is_ok_and(|s| s == "1");

    info!("Adding {count} ui shapes (interleave: {interleave})");

    commands.spawn(Camera2d);

    let side = (count as f32).sqrt().ceil();
    let shape_size = 32.0;
    let spacing = 40.0;

    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        })
        .with_children(|parent| {
            for i in 0..count {
                let (sdf, blend_mode) = match (interleave, i % 4) {
                    (true, 0) => (assets.circle.clone(), BlendMode::Alpha),
                    (true, 1) => (assets.circle.clone(), BlendMode::Additive),
                    (true, 2) => (assets.rectangle.clone(), BlendMode::Alpha),
                    (true, _) => (assets.rectangle.clone(), BlendMode::Additive),
                    (false, _) => (assets.circle.clone(), BlendMode::Alpha),
                };
                let x = (i as f32 % side) * spacing;
                let y = (i as f32 / side).floor() * spacing;

                parent.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(x),
                        top: Val::Px(y),
                        width: Val::Px(shape_size),
                        height: Val::Px(shape_size),
                        ..default()
                    },
                    UiShape {
                        sdf,
                        fill: assets.simple_fill.clone(),
                        blend_mode,
                        ..default()
                    },
                ));
            }
        });
}
