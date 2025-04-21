// Auth0 kimlik doğrulama middleware'i
const { checkJwt, getUserFromAuth0 } = require('./auth0');

const auth = (req, res, next) => {
    // Auth0 middleware'lerini çağır
    checkJwt(req, res, (err) => {
        if (err) {
            return res.status(401).json({
                message: 'Kimlik doğrulama başarısız',
                error: err.message
            });
        }
        
        // Token doğrulandıysa kullanıcı bilgilerini al
        getUserFromAuth0(req, res, next);
    });
};

module.exports = auth;
