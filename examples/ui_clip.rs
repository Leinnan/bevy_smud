//! Demonstrates `UiShape` clipping (via `Overflow`) and the `aux`/`aux_color`
//! fill inputs.
//!
//! The blue container clips the circles that stick out of it, while the green
//! shape outside the container is unaffected.

use bevy::{
    color::palettes::css,
    prelude::*,
    ui::{BackgroundColor, Overflow},
};
use bevy_smud::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, SmudPlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut shaders: ResMut<Assets<Shader>>,
    asset_server: Res<AssetServer>,
) {
    commands.spawn(Camera2d);

    let circle_sdf = asset_server.load("shapes/circle.wesl");

    // A fill that blends from `color` to `aux_color` based on `aux.x`, using
    // the extra fill inputs
    let two_tone_fill = shaders.add_fill_body(
        r#"
    let t = clamp(input.aux.x, 0., 1.);
    let a = smud::sd_fill_alpha_fwidth(input.distance);
    let rgb = mix(input.color.rgb, input.aux_color.rgb, t);
    return vec4<f32>(rgb, a * input.color.a);
    "#,
    );

    // A container that clips its overflowing children
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(25.),
                top: Val::Percent(20.),
                width: Val::Percent(50.),
                height: Val::Percent(60.),
                overflow: Overflow::clip_y(),
                border_radius: BorderRadius::all(Val::Px(24.)),
                ..default()
            },
            BackgroundColor(css::NAVY.into()),
        ))
        .with_children(|parent| {
            // these circles stick out of the container and get clipped
            for i in 0..8u32 {
                let t = i as f32 / 7.;
                parent.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(40. + i as f32 * 85.),
                        top: Val::Px(-60.),
                        width: Val::Px(180.),
                        height: Val::Px(180.),
                        ..default()
                    },
                    UiShape {
                        color: Color::srgb(1., 0.5, 0.2),
                        sdf: circle_sdf.clone(),
                        fill: two_tone_fill.clone(),
                        params: Vec4::new(90., 0., 0., 0.),
                        aux: Vec4::new(t, 0., 0., 0.),
                        aux_color: Color::srgb(0.3, 0.7, 1.),
                        ..default()
                    },
                ));
            }

            // this one sticks out at the bottom
            parent.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(240.),
                    top: Val::Percent(80.),
                    width: Val::Px(180.),
                    height: Val::Px(180.),
                    ..default()
                },
                UiShape {
                    color: Color::srgb(0.2, 1., 0.6),
                    sdf: circle_sdf,
                    params: Vec4::new(90., 0., 0., 0.),
                    ..default()
                },
            ));
        });

    // A shape outside the container is not clipped
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(5.),
            top: Val::Percent(30.),
            width: Val::Px(150.),
            height: Val::Px(150.),
            ..default()
        },
        UiShape {
            color: css::YELLOW.into(),
            sdf: asset_server.load("shapes/rectangle.wesl"),
            params: Vec4::new(75., 75., 0., 0.),
            ..default()
        },
    ));
}
