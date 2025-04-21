const express = require('express');
const router = express.Router();
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const { checkJwt, getUserFromAuth0 } = require('../middleware/auth0');

// Google Maps API anahtarı
const GOOGLE_MAPS_API_KEY = process.env.GOOGLE_MAPS_API_KEY;

// Uploads dizini oluştur (eğer yoksa)
const uploadsDir = path.join(__dirname, '..', 'uploads');
fs.mkdir(uploadsDir, { recursive: true }).catch(console.error);

// Test endpoint'i
router.get('/test', (req, res) => {
  res.json({ message: 'Google Maps API çalışıyor!' });
});

// Google Maps statik görüntü al
router.post('/static-image', async (req, res) => {
  const { center, zoom, size, mapType, bounds } = req.body;
  
  console.log("Google Maps API isteği alındı:", { center, zoom, size, mapType, bounds });
  
  if (!center || !zoom || !size) {
    return res.status(400).json({ error: 'Eksik parametreler' });
  }
  
  try {
    // Görüntü için benzersiz bir isim oluştur
    const timestamp = Date.now();
    const randomString = crypto.randomBytes(8).toString('hex');
    const fileName = `google_maps_${timestamp}_${randomString}.jpg`;
    const filePath = path.join(uploadsDir, fileName);
    
    // Google Maps Statik API'den görüntü al
    let url = `https://maps.googleapis.com/maps/api/staticmap?center=${center.lat},${center.lng}&zoom=${zoom}&size=${size.width}x${size.height}&maptype=${mapType || 'satellite'}&key=${GOOGLE_MAPS_API_KEY}`;
    
    // Eğer sınırlar belirtilmişse, merkez yerine sınırları kullan
    if (bounds) {
      url = `https://maps.googleapis.com/maps/api/staticmap?size=${size.width}x${size.height}&maptype=${mapType || 'satellite'}&visible=${bounds.south},${bounds.west}|${bounds.north},${bounds.east}&key=${GOOGLE_MAPS_API_KEY}`;
    }
    
    console.log("Google Maps API URL:", url);
    
    const response = await axios({
      method: 'get',
      url: url,
      responseType: 'arraybuffer',
      validateStatus: status => status < 500 // 4xx hataları da kabul et
    });
    
    // Eğer hata varsa
    if (response.status !== 200) {
      console.error("Google Maps API hatası:", response.status, response.statusText);
      return res.status(response.status).json({ 
        error: 'Google Maps API hatası', 
        details: response.statusText 
      });
    }
    
    // Görüntüyü kaydet
    await fs.writeFile(filePath, response.data);
    
    // Görüntü URL'sini ve metadata'yı döndür
    const imageUrl = `/uploads/${fileName}`;
    
    res.json({
      imageUrl: imageUrl,
      metadata: {
        source: 'google_maps',
        mapType: mapType || 'satellite',
        zoom: zoom,
        center: center,
        bounds: bounds,
        capturedAt: new Date().toISOString(),
        fileName: fileName
      }
    });
    
  } catch (error) {
    console.error('Google Maps görüntüsü alınırken hata:', error);
    res.status(500).json({ 
      error: 'Görüntü alınırken bir hata oluştu', 
      details: error.message 
    });
  }
});

// Kullanılabilir harita tiplerini getir
router.get('/map-types', (req, res) => {
  const mapTypes = [
    { value: 'roadmap', label: 'Yol Haritası' },
    { value: 'satellite', label: 'Uydu Görüntüsü' },
    { value: 'terrain', label: 'Arazi' },
    { value: 'hybrid', label: 'Hibrit' }
  ];
  
  res.json({ mapTypes });
});

module.exports = router;
