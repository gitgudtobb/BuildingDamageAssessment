const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const Analysis = require('../models/Analysis');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

// AI model service URL
const AI_MODEL_URL = 'http://localhost:5002/api/analyze';

// Dosya yükleme konfigürasyonu
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({
    storage: storage,
    limits: {
        fileSize: 10 * 1024 * 1024 // 10MB limit
    },
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Sadece resim dosyaları yüklenebilir!'));
        }
    }
});

// Test endpoint
router.get('/test', (req, res) => {
    res.json({ message: 'Analysis routes are working!' });
});

// Kullanıcının kendi analizlerini getir
router.get('/my', async (req, res) => {
    try {
        // Sabit bir kullanıcı ID'si kullan
        const defaultUserId = "65c8a5e8b6a1b86d27a93a1b";
        
        console.log("Kullanıcı ID'si:", defaultUserId);
        
        // Kullanıcının analizlerini bul
        const analyses = await Analysis.find({ user: defaultUserId })
            .sort({ createdAt: -1 })
            .limit(10);
        
        res.json({ analyses });
    } catch (error) {
        console.error('Analizleri getirme hatası:', error);
        res.status(500).json({ error: 'Analizleri getirirken bir hata oluştu' });
    }
});

// Analiz oluştur
router.post('/', async (req, res) => {
  try {
    console.log("Analiz isteği alındı:", req.body);
    
    // Sabit bir kullanıcı ID'si kullan
    const defaultUserId = "65c8a5e8b6a1b86d27a93a1b";
    
    console.log("Analiz oluşturuluyor, kullanıcı ID:", defaultUserId);
    
    // Şu an için yapay zeka entegrasyonu olmadığı için dummy sonuçlar dönelim
    const results = {
      damageLevel: "moderate",
      damagePercentage: 45,
      affectedAreas: ["roof", "walls", "windows"],
      recommendations: [
        "Çatıdaki hasarlar onarılmalı",
        "Duvar çatlakları kontrol edilmeli",
        "Kırık pencereler değiştirilmeli"
      ],
      riskAssessment: "medium",
      estimatedRepairCost: {
        min: 15000,
        max: 25000,
        currency: "TRY"
      }
    };
    
    // Yeni analiz oluştur
    const newAnalysis = new Analysis({
      user: defaultUserId, // Mevcut kullanıcı ID'si
      location: {
        type: "Point",
        coordinates: [29.0335, 41.0053], // İstanbul koordinatları
        address: "İstanbul, Türkiye"
      },
      results: results,
      metadata: {
        buildingType: "Residential",
        constructionYear: 2000,
        floorCount: 5
      },
      images: {
        before: req.body.beforeImage || "https://example.com/before.jpg",
        after: req.body.afterImage || "https://example.com/after.jpg"
      },
      createdAt: new Date(),
      status: "completed"
    });
    
    await newAnalysis.save();
    
    res.status(201).json({
      message: 'Analiz başarıyla kaydedildi',
      analysisId: newAnalysis._id,
      results: results
    });
  } catch (error) {
    console.error('Analiz oluşturma hatası:', error);
    res.status(500).json({ error: 'Analiz oluşturulurken bir hata oluştu', details: error.message });
  }
});

// Belirli bir analizi getir
router.get('/:id', async (req, res) => {
    try {
        // Sabit bir kullanıcı ID'si kullan
        const defaultUserId = "65c8a5e8b6a1b86d27a93a1b";
        
        const analysis = await Analysis.findOne({
            _id: req.params.id,
            user: defaultUserId
        });
        
        if (!analysis) {
            return res.status(404).json({ error: 'Analiz bulunamadı veya bu analizi görüntüleme yetkiniz yok' });
        }
        
        res.json(analysis);
    } catch (error) {
        console.error('Analiz getirme hatası:', error);
        res.status(500).json({ error: 'Analiz getirilirken bir hata oluştu' });
    }
});

// Konum bazlı analiz arama
router.get('/location/nearby', async (req, res) => {
    try {
        const { longitude, latitude, maxDistance = 1000 } = req.query;
        
        // Sabit bir kullanıcı ID'si kullan
        const defaultUserId = "65c8a5e8b6a1b86d27a93a1b";

        const analyses = await Analysis.find({
            user: defaultUserId,
            location: {
                $near: {
                    $geometry: {
                        type: 'Point',
                        coordinates: [parseFloat(longitude), parseFloat(latitude)]
                    },
                    $maxDistance: parseInt(maxDistance)
                }
            }
        }).limit(20);

        res.json(analyses);
    } catch (error) {
        res.status(500).json({ error: 'Yakındaki analizler aranırken bir hata oluştu' });
    }
});

// Get all analyses for the authenticated user
router.get('/analyses', async (req, res) => {
    try {
      // For now, return a mock response since we don't have a database set up yet
      const mockAnalyses = [
        {
          id: '1',
          beforeImageUrl: 'https://via.placeholder.com/500x300?text=Before+Image',
          afterImageUrl: 'https://via.placeholder.com/500x300?text=After+Image',
          location: {
            type: 'Point',
            coordinates: [29.0335, 41.0053]
          },
          metadata: {
            buildingType: 'Residential',
            constructionYear: 2010,
            floorCount: 5
          },
          results: {
            damagePercentage: 65,
            recommendations: [
              'Bina yapısal değerlendirme gerektirir',
              'Acil güçlendirme çalışması önerilir'
            ],
            details: {
              'Yapısal Hasar': 'Ağır',
              'Duvar Çatlakları': 'Yaygın',
              'Çatı Hasarı': 'Orta',
              'Zemin Oturması': 'Var'
            }
          },
          createdAt: new Date().toISOString()
        },
        {
          id: '2',
          beforeImageUrl: 'https://via.placeholder.com/500x300?text=Before+Image+2',
          afterImageUrl: 'https://via.placeholder.com/500x300?text=After+Image+2',
          location: {
            type: 'Point',
            coordinates: [29.1335, 41.1053]
          },
          metadata: {
            buildingType: 'Commercial',
            constructionYear: 2015,
            floorCount: 3
          },
          results: {
            damagePercentage: 25,
            recommendations: [
              'Hafif onarım çalışması önerilir',
              'Düzenli kontrol yapılmalıdır'
            ],
            details: {
              'Yapısal Hasar': 'Hafif',
              'Duvar Çatlakları': 'Az',
              'Çatı Hasarı': 'Minimal',
              'Zemin Oturması': 'Yok'
            }
          },
          createdAt: new Date(Date.now() - 86400000).toISOString() // 1 day ago
        }
      ];
      
      res.json(mockAnalyses);
    } catch (error) {
      console.error('Error fetching analyses:', error);
      res.status(500).json({ message: 'Analizler yüklenirken bir hata oluştu' });
    }
  });

module.exports = router;
