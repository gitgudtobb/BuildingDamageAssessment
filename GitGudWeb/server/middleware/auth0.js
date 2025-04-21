const { expressjwt: jwt } = require('express-jwt');
const jwksRsa = require('jwks-rsa');
const User = require('../models/User');

// Auth0 middleware - token doğrulama
const checkJwt = jwt({
  secret: jwksRsa.expressJwtSecret({
    cache: true,
    rateLimit: true,
    jwksRequestsPerMinute: 5,
    jwksUri: `https://${process.env.AUTH0_DOMAIN}/.well-known/jwks.json`
  }),
  audience: process.env.AUTH0_AUDIENCE,
  issuer: `https://${process.env.AUTH0_DOMAIN}/`,
  algorithms: ['RS256'],
  credentialsRequired: true // Token zorunlu
});

// Kullanıcı bilgilerini MongoDB'den alıp req.user'a ekleyen middleware
const getUserFromAuth0 = async (req, res, next) => {
  try {
    console.log('Auth0 middleware çalışıyor...');
    
    // Token yoksa veya geçersizse, hata döndür
    if (!req.auth || !req.auth.sub) {
      console.log('Token yok veya geçersiz!');
      return res.status(401).json({
        message: 'Kimlik doğrulama başarısız',
        error: 'Geçersiz token'
      });
    }

    // Eğer zaten bu istek için kullanıcı bilgisi işlendiyse, tekrar işleme
    if (req.user) {
      console.log('Kullanıcı zaten işlenmiş, middleware atlanıyor');
      return next();
    }

    console.log('Token bilgisi:', 'Token var');
    console.log('req.auth:', JSON.stringify(req.auth, null, 2));

    const auth0Id = req.auth.sub;
    console.log('Auth0 ID:', auth0Id);
    
    // Auth0 ID'sine göre kullanıcıyı bul
    let user = await User.findOne({ auth0Id });
    console.log('MongoDB\'de kullanıcı bulundu mu?', user ? 'Evet' : 'Hayır');
    
    // Kullanıcı yoksa yeni oluştur
    if (!user) {
      console.log('Kullanıcı bulunamadı, yeni kullanıcı oluşturuluyor...');
      
      // Önce email ile kontrol et (eğer email varsa)
      if (req.auth.email) {
        const existingUser = await User.findOne({ email: req.auth.email });
        if (existingUser) {
          console.log('Email ile kullanıcı bulundu, auth0Id güncelleniyor...');
          existingUser.auth0Id = auth0Id;
          existingUser.authType = 'auth0';
          
          // Auth0 profilinden gelen ek bilgileri güncelle
          if (req.auth.nickname) existingUser.username = req.auth.nickname;
          if (req.auth.name) existingUser.name = req.auth.name;
          if (req.auth.picture) existingUser.profilePicture = req.auth.picture;
          
          await existingUser.save();
          user = existingUser;
          console.log('Kullanıcı güncellendi:', user._id);
        }
      }
      
      // Hala kullanıcı bulunamadıysa yeni oluştur
      if (!user) {
        // Auth0 profilinden kullanıcı adı ve email bilgilerini al
        const nickname = req.auth.nickname || '';
        const email = req.auth.email || '';
        const username = nickname || (email ? email.split('@')[0] : `user_${Date.now()}`);
        
        console.log('Yeni kullanıcı için bilgiler:', {
          username,
          email,
          auth0Id
        });
        
        const newUserData = {
          auth0Id,
          email: email,
          username: username,
          name: req.auth.name || '',
          profilePicture: req.auth.picture || '',
          authType: 'auth0'
        };
        
        try {
          user = await User.create(newUserData);
          console.log('Yeni kullanıcı oluşturuldu:', user._id);
        } catch (error) {
          console.error('Kullanıcı oluşturma hatası:', error);
          
          // Eğer username veya email çakışması varsa, benzersiz değerler oluştur
          if (error.code === 11000) {
            console.log('Benzersiz alan hatası, yeni değerler oluşturuluyor...');
            newUserData.username = `${username}_${Date.now()}`;
            
            if (error.keyPattern && error.keyPattern.email) {
              newUserData.email = `auth0_${Date.now()}@example.com`;
            }
            
            try {
              user = await User.create(newUserData);
              console.log('Benzersiz değerlerle kullanıcı oluşturuldu:', user._id);
            } catch (secondError) {
              console.error('İkinci kullanıcı oluşturma denemesi hatası:', secondError);
              return next(secondError);
            }
          } else if (error.name === 'ValidationError') {
            // Validation error handling - ensure email is set
            console.log('Doğrulama hatası, gerekli alanlar düzeltiliyor...');
            
            // Email validation error - set a temporary email if not present
            if (error.errors && error.errors.email) {
              newUserData.email = `auth0_${Date.now()}@example.com`;
              console.log('Geçici email oluşturuldu:', newUserData.email);
            }
            
            // Try creating user again with fixed data
            try {
              user = await User.create(newUserData);
              console.log('Düzeltilmiş verilerle kullanıcı oluşturuldu:', user._id);
            } catch (thirdError) {
              console.error('Üçüncü kullanıcı oluşturma denemesi hatası:', thirdError);
              return next(thirdError);
            }
          } else {
            return next(error); // Başka bir hata varsa yeniden fırlat
          }
        }
      }
    } else {
      // Kullanıcı varsa, Auth0 profilinden gelen bilgilerle güncelle
      console.log('Kullanıcı bulundu, profil bilgileri güncelleniyor...');
      let isModified = false;
      
      if (req.auth.nickname && user.username !== req.auth.nickname) {
        user.username = req.auth.nickname;
        isModified = true;
      }
      
      if (req.auth.name && user.name !== req.auth.name) {
        user.name = req.auth.name;
        isModified = true;
      }
      
      if (req.auth.picture && user.profilePicture !== req.auth.picture) {
        user.profilePicture = req.auth.picture;
        isModified = true;
      }
      
      // Email güncellemesini kaldırıyoruz - Auth0 ile senkronizasyon sorunlarını önlemek için
      // Email sadece ilk kayıt sırasında alınacak, sonradan güncellenmeyecek
      
      // Değişiklik varsa kaydet
      if (isModified) {
        try {
          await user.save();
          console.log('Kullanıcı profili güncellendi');
        } catch (error) {
          console.error('Kullanıcı güncelleme hatası:', error);
          // Güncelleme hatası olsa bile devam et
        }
      }
    }
    
    if (user) {
      req.user = user;
      console.log('req.user ayarlandı:', req.user._id);
    } else {
      console.log('Kullanıcı bulunamadı veya oluşturulamadı!');
      return res.status(404).json({
        message: 'Kullanıcı bulunamadı veya oluşturulamadı',
        error: 'User not found'
      });
    }
    
    next();
  } catch (error) {
    console.error('Auth0 user processing error:', error);
    next(error);
  }
};

module.exports = {
  checkJwt,
  getUserFromAuth0
};
