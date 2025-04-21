const express = require('express');
const router = express.Router();
const path = require('path');
const ee = require('@google/earthengine');
const axios = require('axios');
const fs = require('fs').promises;
const crypto = require('crypto');

// Define API base URL
const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:5001';

// Get the absolute path to the credentials file
const credentialsPath = path.join(
  __dirname,
  '..',
  'config',
  'earth-engine-credentials.json'
);
const privateKey = require(credentialsPath);

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, '..', 'uploads');
fs.mkdir(uploadsDir, { recursive: true }).catch(console.error);

// Initialize Earth Engine with more detailed logging
const initializeEarthEngine = async (context) => {
  return new Promise((resolve, reject) => {
    try {
      console.log('Starting Earth Engine initialization...');
      console.log('Using credentials from:', credentialsPath);

      ee.data.authenticateViaPrivateKey(
        privateKey,
        () => {
          console.log(
            'Authentication successful, initializing Earth Engine...'
          );
          ee.initialize(
            null,
            null,
            () => {
              console.log('Earth Engine initialized successfully');
              resolve();
            },
            (err) => {
              console.error('Error initializing Earth Engine:', err);
              reject(err);
            }
          );
        },
        (err) => {
          console.error('Error authenticating with Earth Engine:', err);
          reject(err);
        }
      );
    } catch (error) {
      console.error('Error in Earth Engine initialization:', error);
      reject(error);
    }
  });
};

// Tarih önbelleği
const dateCache = new Map();
const CACHE_EXPIRY = 5 * 60 * 1000; // 5 dakika

// Bounds için anahtar oluştur
function getBoundsCacheKey(bounds) {
  if (!bounds) return null;
  // Koordinatları 2 ondalık basamağa yuvarla (yaklaşık 1km hassasiyet)
  const north = Math.round(bounds.north * 100) / 100;
  const south = Math.round(bounds.south * 100) / 100;
  const east = Math.round(bounds.east * 100) / 100;
  const west = Math.round(bounds.west * 100) / 100;
  return `${north},${south},${east},${west}`;
}

// Get available dates for a location
router.post('/dates', async (req, res) => {
  const { lat, lng, bounds } = req.body;

  if (!lat || !lng) {
    return res.status(400).json({ error: 'Missing required parameters' });
  }

  try {
    // Önbellekte var mı kontrol et
    const cacheKey = getBoundsCacheKey(bounds);
    if (cacheKey && dateCache.has(cacheKey)) {
      const cached = dateCache.get(cacheKey);
      if (Date.now() - cached.timestamp < CACHE_EXPIRY) {
        console.log('Returning cached dates for bounds:', cacheKey);
        return res.json({ dates: cached.dates });
      } else {
        dateCache.delete(cacheKey);
      }
    }

    await initializeEarthEngine('dates');
    console.log('Fetching dates for location:', { lat, lng, bounds });

    // Bölge geometrisi oluştur
    let region;
    if (bounds) {
      region = ee.Geometry.Rectangle([
        bounds.west,
        bounds.south,
        bounds.east,
        bounds.north,
      ]);
    } else {
      // Eğer bounds yoksa, merkez noktanın etrafında 1km'lik alan al
      const point = ee.Geometry.Point([lng, lat]);
      region = point.buffer(1000);
    }

    // Son 6 aylık görüntüleri sorgula
    const endDate = new Date();
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - 6);

    const collection = ee
      .ImageCollection('GOOGLE/DYNAMICWORLD/V1')
      .filterBounds(region)
      .filterDate(ee.Date(startDate), ee.Date(endDate));

    collection.getInfo((info) => {
      try {
        if (!info || !info.features || info.features.length === 0) {
          // DynamicWorld'de görüntü bulunamazsa Sentinel-2'ye bak
          const s2Collection = ee
            .ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(ee.Date(startDate), ee.Date(endDate))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .sort('CLOUDY_PIXEL_PERCENTAGE');

          s2Collection.getInfo((s2Info) => {
            processDates(s2Info, res, cacheKey);
          });
          return;
        }
        processDates(info, res, cacheKey);
      } catch (error) {
        console.error('Error processing collection:', error);
        res.status(500).json({ error: 'Error processing dates' });
      }
    });
  } catch (error) {
    console.error('Error in /dates endpoint:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

function processDates(info, res, cacheKey) {
  if (!info || !info.features || info.features.length === 0) {
    return res
      .status(404)
      .json({ error: 'No images found for the specified location' });
  }

  console.log('Processing', info.features.length, 'images...');

  const dates = [
    ...new Set(
      info.features.map((f) => {
        const timestamp = f.properties['system:time_start'];
        return new Date(timestamp).toISOString().split('T')[0];
      })
    ),
  ].sort();

  console.log('Found', dates.length, 'unique dates');

  // Sonuçları önbelleğe al
  if (cacheKey) {
    dateCache.set(cacheKey, {
      dates: dates,
      timestamp: Date.now(),
    });
  }

  res.json({ dates: dates });
}

// Get available dates for a specific location
router.post('/dates', async (req, res) => {
  const { lat, lng, bounds } = req.body;

  if (!lat || !lng || !bounds) {
    return res.status(400).json({
      error: 'Missing required parameters',
      received: { lat, lng, bounds },
    });
  }

  try {
    await initializeEarthEngine('dates');

    console.log('Fetching available dates for:', { lat, lng, bounds });

    // Create region from bounds
    const region = ee.Geometry.Rectangle([
      bounds.west,
      bounds.south,
      bounds.east,
      bounds.north,
    ]);

    // Son 6 aylık görüntüleri sorgula
    const endDate = new Date();
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - 6);

    // Önce DynamicWorld koleksiyonunu dene
    const collection = ee
      .ImageCollection('GOOGLE/DYNAMICWORLD/V1')
      .filterBounds(region)
      .filterDate(ee.Date(startDate), ee.Date(endDate));

    collection.getInfo((info) => {
      try {
        let dates = [];

        if (info && info.features && info.features.length > 0) {
          dates = info.features.map((f) => f.properties['system:time_start']);
        }

        // Sentinel-2'den de tarihleri al
        const s2Collection = ee
          .ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(region)
          .filterDate(ee.Date(startDate), ee.Date(endDate))
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
          .distinct('system:time_start');

        s2Collection.getInfo((s2Info) => {
          if (s2Info && s2Info.features && s2Info.features.length > 0) {
            const s2Dates = s2Info.features.map(
              (f) => f.properties['system:time_start']
            );
            dates = [...new Set([...dates, ...s2Dates])].sort((a, b) => b - a); // En yeni tarihler başta
          }

          if (dates.length === 0) {
            return res.status(404).json({
              error: 'Bu bölge için kullanılabilir görüntü bulunamadı',
            });
          }

          res.json({
            dates: dates.map((timestamp) => new Date(timestamp).toISOString()),
          });
        });
      } catch (error) {
        console.error('Error processing collection:', error);
        res.status(500).json({ error: 'Error processing satellite imagery' });
      }
    });
  } catch (error) {
    console.error('Error in /dates endpoint:', error);
    res.status(500).json({
      error: 'Internal server error',
      details: error.message,
    });
  }
});

// Get satellite image for a specific date and location
router.post('/image', async (req, res) => {
  const { lat, lng, date, bounds, viewport } = req.body;

  try {
    await initializeEarthEngine('image');

    const region = ee.Geometry.Rectangle([
      bounds.west,
      bounds.south,
      bounds.east,
      bounds.north,
    ]);

    const targetDate = new Date(date);
    const startDate = new Date(targetDate);
    startDate.setDate(startDate.getDate() - 5);
    const endDate = new Date(targetDate);
    endDate.setDate(endDate.getDate() + 5);

    // Sentinel-2 koleksiyonunu kullan
    const collection = ee
      .ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(region)
      .filterDate(ee.Date(startDate), ee.Date(endDate))
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
      .sort('CLOUDY_PIXEL_PERCENTAGE'); // En az bulutlu olanı seç

    collection.getInfo(async (info) => {
      try {
        if (!info || !info.features || !info.features.length) {
          return res
            .status(404)
            .json({ error: 'Bu tarih için uygun görüntü bulunamadı' });
        }

        const feature = info.features[0];
        const image = ee.Image(feature.id);

        // RGB bantlarını seç ve görselleştirme parametrelerini uygula
        const visualizedImage = image
          .select(['B4', 'B3', 'B2'])
          .visualize({
            min: 0,
            max: 3000,
            gamma: 1.2
          });

        // Thumbnail boyutlarını sınırla
        const maxDimension = 1024; // Maximum boyut
        const aspectRatio = viewport.width / viewport.height;
        let width, height;
        
        if (aspectRatio > 1) {
          width = maxDimension;
          height = Math.round(maxDimension / aspectRatio);
        } else {
          height = maxDimension;
          width = Math.round(maxDimension * aspectRatio);
        }

        const thumbnail = visualizedImage.getThumbURL({
          dimensions: `${width}x${height}`,
          region: region,
          format: 'jpg',
          quality: 85
        });

        res.json({
          imageUrl: `${API_BASE_URL}/api/earth-engine/proxy-image?url=${encodeURIComponent(
            thumbnail
          )}`,
          metadata: {
            datetime: new Date(
              feature.properties['system:time_start']
            ).toISOString(),
            cloudCover: feature.properties.CLOUDY_PIXEL_PERCENTAGE || 0,
            resolution: `${width}x${height}`,
            source: 'Sentinel-2',
            spatialResolution: '10m'
          }
        });
      } catch (error) {
        console.error('Error processing image:', error);
        res.status(500).json({ error: 'Error processing satellite imagery' });
      }
    });
  } catch (error) {
    console.error('Error in /image endpoint:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

function getOptimizedParameters(zoomLevel, source) {
  const baseParams = {
    min: 0,
    max: 2000,
    gamma: 1.2,
    saturation: 1.3,
    brightnessFactor: 1.2,
    sharpnessFactor: 0.3,
    contrastRadius: 2,
  };

  if (zoomLevel >= 19) {
    return {
      ...baseParams,
      min: 100,
      max: 2500,
      gamma: 1.1,
      brightnessFactor: 1.4,
      sharpnessFactor: 0.4,
      contrastRadius: 1,
      superResScale: 2,
    };
  } else if (zoomLevel >= 17) {
    return {
      ...baseParams,
      brightnessFactor: 1.3,
      sharpnessFactor: 0.35,
      superResScale: 1.5,
    };
  }
  return baseParams;
}

function applySuperResolution(image, params) {
  // Bicubic yeniden örnekleme ile super-resolution
  const upscaled = image.resample('bicubic');

  // Detay geliştirme
  const enhanced = upscaled
    .convolve(ee.Kernel.gaussian(3, 1.5, 'pixels'))
    .add(
      upscaled.convolve(ee.Kernel.laplacian8()).multiply(params.sharpnessFactor)
    );

  return enhanced;
}

// Proxy and cache Earth Engine images
router.get('/proxy-image', async (req, res) => {
  const { url } = req.query;

  if (!url) {
    return res.status(400).json({ error: 'URL parameter is required' });
  }

  console.log('Proxying image from URL:', url);
  try {
    // Generate a unique filename based on the URL
    const hash = crypto.createHash('md5').update(url).digest('hex');
    const filename = `${hash}.jpg`;
    const filepath = path.join(uploadsDir, filename);

    // Check if file already exists
    try {
      await fs.access(filepath);
      console.log('Image already exists, serving from cache:', filepath);
      return res.sendFile(filepath);
    } catch (err) {
      // File doesn't exist, continue with download
    }

    // Initialize Earth Engine and get fresh token
    await initializeEarthEngine('proxy');
    const authClient = ee.data.getAuthToken();

    // Remove 'Bearer ' prefix if it exists
    const token = authClient.replace(/^Bearer\s+/i, '');

    console.log('Making authenticated request to Earth Engine API...');
    const response = await axios({
      method: 'get',
      url: url,
      responseType: 'arraybuffer',
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'image/jpeg,image/png,image/*',
        'User-Agent': 'Mozilla/5.0',
      },
      maxRedirects: 5,
      validateStatus: (status) => status >= 200 && status < 300,
      timeout: 30000, // 30 second timeout
    });

    if (!response.data || response.data.length === 0) {
      throw new Error('No image data received');
    }

    console.log('Received image data, size:', response.data.length, 'bytes');
    console.log('Response headers:', response.headers);

    // Save the file
    await fs.writeFile(filepath, response.data);
    console.log('Image saved to:', filepath);

    // Set appropriate headers
    res.setHeader(
      'Content-Type',
      response.headers['content-type'] || 'image/jpeg'
    );
    res.setHeader('Cache-Control', 'public, max-age=31536000');
    res.setHeader('Access-Control-Allow-Origin', '*');

    // Send the file
    res.sendFile(filepath);
  } catch (error) {
    console.error('Error proxying image:', error);
    if (error.response) {
      console.error('Error response:', {
        status: error.response.status,
        statusText: error.response.statusText,
        headers: error.response.headers,
      });
    }
    res.status(500).json({
      error: 'Failed to proxy image',
      details: error.message,
      url: url,
    });
  }
});

module.exports = router;
