const User = require('../models/User');

// Auth0 kimlik doğrulama için controller fonksiyonları
const auth0Auth = {
  // Auth0 kullanıcı profili
  getProfile: async (req, res) => {
    try {
      console.log('getProfile çalışıyor...');
      console.log('req.user:', req.user ? 'Mevcut' : 'Yok');
      
      if (!req.user) {
        console.log('Kullanıcı bulunamadı!');
        return res.status(404).json({ 
          message: 'Kullanıcı bulunamadı',
          authType: 'auth0'
        });
      }
      
      console.log('Kullanıcı bulundu, ID:', req.user._id);
      
      // Hassas bilgileri çıkararak kullanıcı bilgilerini döndür
      const userObj = req.user.toObject ? req.user.toObject() : req.user;
      const { password, ...userWithoutPassword } = userObj;
      
      res.json({
        ...userWithoutPassword,
        authType: 'auth0'
      });
    } catch (error) {
      console.error('Auth0 profile error:', error);
      res.status(500).json({ 
        message: 'Profil bilgileri alınamadı', 
        error: error.message,
        authType: 'auth0'
      });
    }
  },
  
  // Auth0 kullanıcı profil güncelleme
  updateProfile: async (req, res) => {
    try {
      const userId = req.user._id;
      const { username, name, bio, phone, address } = req.body;
      
      // Email güncellemesini kaldırdık - Auth0 ile senkronizasyon sorunlarını önlemek için
      
      const updatedUser = await User.findByIdAndUpdate(
        userId,
        { 
          username, 
          name, 
          bio, 
          phone, 
          address 
        },
        { new: true, runValidators: true }
      );
      
      if (!updatedUser) {
        return res.status(404).json({ error: 'Kullanıcı bulunamadı' });
      }
      
      res.json({
        _id: updatedUser._id,
        username: updatedUser.username,
        name: updatedUser.name,
        email: updatedUser.email, // Email değiştirilemiyor, sadece gösteriliyor
        picture: updatedUser.profilePicture,
        bio: updatedUser.bio,
        phone: updatedUser.phone,
        address: updatedUser.address,
        createdAt: updatedUser.createdAt
      });
    } catch (error) {
      console.error('Profil güncelleme hatası:', error);
      res.status(500).json({ error: 'Profil güncellenirken bir hata oluştu', details: error.message });
    }
  },
};

module.exports = {
  auth0Auth
};
