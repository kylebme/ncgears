"""Generate and export a smooth two-lobed 1:1 gear pair."""

import ncgears

pair = ncgears.generate(
    "phi - 0.08*sin(2*phi)",
    teeth=24,
    module=1.5,
    name="two_lobe",
)

print(pair.summary())
pair.export_dxf(pair.directory / "two_lobe_pair.dxf")
pair.export_svg(pair.directory / "two_lobe_pair.svg")
