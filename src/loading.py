import cdsapi
import os

dataset = "reanalysis-era5-single-levels"

variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "evaporation",
    "potential_evaporation",
    "sub_surface_runoff",
    "surface_runoff",
    "large_scale_precipitation",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "vertically_integrated_moisture_divergence"
]

start_year = 1990
end_year   = 2026
output_dir = "era5_downloads"

os.makedirs(output_dir, exist_ok=True)
client = cdsapi.Client()

for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        target = os.path.join(output_dir, f"era5_{year}_{month:02d}.zip")

        if os.path.exists(target):
            print(f"Skipping {year}-{month:02d} — already exists.")
            continue

        print(f"Downloading {year}-{month:02d}...")
        request = {
            "product_type": ["reanalysis"],
            "variable": variables,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "data_format": "netcdf",
            "download_format": "zip",
            "area": [15, 0.8, -35, 48]
        }

        try:
            client.retrieve(dataset, request).download(target)
            print(f"  Saved to {target}")
        except Exception as e:
            print(f"  ERROR {year}-{month:02d}: {e}")
            if os.path.exists(target):
                os.remove(target)  # remove partial file

print("Loop complete.")