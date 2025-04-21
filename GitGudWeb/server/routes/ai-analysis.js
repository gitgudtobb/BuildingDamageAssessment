const express = require('express');
const router = express.Router();
const axios = require('axios');

// Python AI API URL
const AI_API_URL = 'http://localhost:5002';

// Hasar analizi endpoint'i
router.post('/damage-analysis', async (req, res) => {
    try {
        console.log('Hasar analizi isteği alındı');
        
        // İstemciden gelen görüntüleri al
        const { preImage, postImage } = req.body;
        
        if (!preImage || !postImage) {
            return res.status(400).json({ 
                error: 'Afet öncesi ve sonrası görüntüler gereklidir' 
            });
        }
        
        // Python API'sine istek gönder
        const response = await axios.post(`${AI_API_URL}/analyze`, {
            preImage,
            postImage
        });
        
        // Sonuçları döndür
        return res.json(response.data);
    } catch (error) {
        console.error('Hasar analizi hatası:', error.message);
        
        // API'ye bağlanılamadıysa özel hata mesajı
        if (error.code === 'ECONNREFUSED') {
            return res.status(503).json({ 
                error: 'AI servisi şu anda çalışmıyor. Lütfen Python API\'sinin çalıştığından emin olun.' 
            });
        }
        
        // Diğer hatalar için genel hata mesajı
        return res.status(500).json({ 
            error: 'Hasar analizi sırasında bir hata oluştu',
            details: error.message
        });
    }
});

// AI sağlık kontrolü
router.get('/health', async (req, res) => {
    try {
        const response = await axios.get(`${AI_API_URL}/health`);
        return res.json(response.data);
    } catch (error) {
        console.error('AI sağlık kontrolü hatası:', error.message);
        return res.status(503).json({ 
            status: 'unhealthy',
            error: 'AI servisi şu anda çalışmıyor'
        });
    }
});

module.exports = router;
