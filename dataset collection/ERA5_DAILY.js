// =================== CONFIGURATION ===================
var start = ee.Date('2016-04-01');
var end = ee.Date('2016-05-30');

var uttarakhand = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
  .filter(ee.Filter.eq('ADM1_NAME', 'Uttarakhand'));

// Load ERA5 dataset
var era5 = ee.ImageCollection('ECMWF/ERA5/DAILY')
  .filterDate(start, end)
  .select([
    'mean_2m_air_temperature',
    'dewpoint_2m_temperature',
    'total_precipitation',
    'u_component_of_wind_10m',
    'v_component_of_wind_10m'
  ]);

// Get number of days in range
var nDays = end.difference(start, 'day');

// Loop over each day
ee.List.sequence(0, nDays.subtract(1)).getInfo().forEach(function(dayOffset) {
  var date = start.advance(dayOffset, 'day');
  var dateStr = date.format('YYYY_MM_dd').getInfo();

  var dailyImage = era5.filterDate(date, date.advance(1, 'day')).first();

  // Only proceed if image exists
  if (dailyImage) {
    var t2m = dailyImage.select('mean_2m_air_temperature').subtract(273.15).rename('t2m_C').toFloat();
    var d2m = dailyImage.select('dewpoint_2m_temperature').subtract(273.15).rename('d2m_C').toFloat();
    var tp = dailyImage.select('total_precipitation').multiply(1000).rename('precip_mm').toFloat();
    var u10 = dailyImage.select('u_component_of_wind_10m').rename('u10').toFloat();
    var v10 = dailyImage.select('v_component_of_wind_10m').rename('v10').toFloat();

    var stacked = t2m.addBands([d2m, tp, u10, v10]).clip(uttarakhand);

    Export.image.toDrive({
      image: stacked,
      description: 'ERA5_' + dateStr,
      folder: 'GEE_Exports',
      fileNamePrefix: 'era5_' + dateStr,
      region: uttarakhand.geometry(),
      scale: 10000,
      crs: 'EPSG:4326',
      maxPixels: 1e13
    });
  }
});

// =================== VISUALIZATION ===================
Map.centerObject(uttarakhand, 7);
