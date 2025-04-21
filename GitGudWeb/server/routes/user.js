const express = require('express');
const router = express.Router();
const User = require('../models/User');
const { checkJwt, getUserFromAuth0 } = require('../middleware/auth0');
const { auth0Auth } = require('../controllers/authController');

// Auth0 kullanıcı profili bilgilerini getir - Auth0 middleware olmadan
router.get('/profile', async (req, res) => {
    try {
        console.log("getProfile çalışıyor...");
        
        // Sabit bir kullanıcı ID'si kullan
        const defaultUserId = "65c8a5e8b6a1b86d27a93a1b";
        let user = await User.findById(defaultUserId);
        
        // Eğer kullanıcı yoksa oluştur
        if (!user) {
            console.log("Kullanıcı bulunamadı, yeni oluşturuluyor...");
            user = new User({
                _id: defaultUserId,
                username: "demo_user",
                email: "demo@example.com",
                name: "Demo Kullanıcı",
                profilePicture: "https://ui-avatars.com/api/?name=Demo+User&background=random",
                bio: "Bu bir demo kullanıcısıdır",
                phone: "+90 555 123 4567",
                address: "İstanbul, Türkiye",
                authType: "auth0"
            });
            
            await user.save();
            console.log("Demo kullanıcısı oluşturuldu:", user._id);
        } else {
            console.log("Demo kullanıcısı bulundu:", user._id);
        }
        
        if (!user) {
            return res.status(404).json({ error: 'Kullanıcı bulunamadı' });
        }
        
        console.log("Kullanıcı bilgileri gönderiliyor, ID:", user._id);
        
        res.json({
            _id: user._id,
            username: user.username,
            name: user.name,
            email: user.email,
            picture: user.profilePicture,
            bio: user.bio,
            phone: user.phone,
            address: user.address,
            createdAt: user.createdAt
        });
    } catch (error) {
        console.error('Profil bilgisi getirme hatası:', error);
        res.status(500).json({ error: 'Profil bilgisi getirilirken bir hata oluştu', details: error.message });
    }
});

// Kullanıcı profil bilgilerini güncelle
router.put('/profile', checkJwt, getUserFromAuth0, auth0Auth.updateProfile);

// Tüm kullanıcıları listele (sadece admin için)
router.get('/all', checkJwt, getUserFromAuth0, async (req, res) => {
  try {
    // Kullanıcının admin olup olmadığını kontrol et
    if (!req.user || !req.user.roles.includes('admin')) {
      return res.status(403).json({ 
        message: 'Bu işlem için yetkiniz bulunmamaktadır'
      });
    }
    
    const users = await User.find().select('-password');
    res.json(users);
  } catch (error) {
    console.error('Users list error:', error);
    res.status(500).json({ message: 'Kullanıcılar listelenemedi', error: error.message });
  }
});

module.exports = router;
