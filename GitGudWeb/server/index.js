const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const cookieParser = require('cookie-parser');
const morgan = require('morgan');
const helmet = require('helmet');
const path = require('path');
require('dotenv').config();

const app = express();

// CORS ayarları
app.use(cors({
    origin: ['http://localhost:5173', 'http://localhost:5175', 'https://earthengine.googleapis.com', 'https://maps.googleapis.com'],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    exposedHeaders: ['Content-Type', 'Authorization'] 
}));

// Diğer middleware'ler
app.use(helmet({
    crossOriginResourcePolicy: { policy: "cross-origin" },
    crossOriginOpenerPolicy: { policy: "same-origin-allow-popups" },
    contentSecurityPolicy: false
}));
app.use(morgan('dev')); // Loglama için
app.use(cookieParser());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Serve static files from uploads directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Uploads klasörünü oluştur
const fs = require('fs');
if (!fs.existsSync('./uploads')) {
    fs.mkdirSync('./uploads');
}

// MongoDB bağlantısı
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/gitgudweb')
.then(() => console.log('MongoDB bağlantısı başarılı'))
.catch(err => console.error('MongoDB bağlantı hatası:', err));

// Auth0 middleware'ini import et
const { checkJwt, getUserFromAuth0 } = require('./middleware/auth0');

// Auth0 korumalı rotalar için middleware
const auth0Protected = [checkJwt, getUserFromAuth0];

// Geçici kullanıcı middleware'i (Auth0 entegrasyonu tam çalışana kadar)
const tempUserMiddleware = (req, res, next) => {
    // Eğer Auth0 token'ı varsa, onu kullan
    if (req.user) {
        return next();
    }
    
    // Yoksa geçici kullanıcı bilgisi ekle
    req.user = { _id: '65c8a5e8b6a1b86d27a93a1b' };
    next();
};

// Routes
const analysisRoutes = require('./routes/analysis');
const earthEngineRoutes = require('./routes/earth-engine');
const userRoutes = require('./routes/user');
const googleMapsRoutes = require('./routes/google-maps');
const aiAnalysisRoutes = require('./routes/ai-analysis');

// API rotaları
app.use('/api/earth-engine', earthEngineRoutes);
app.use('/api/google-maps', googleMapsRoutes);
app.use('/api/ai-analysis', aiAnalysisRoutes);

// Auth0 korumalı rotalar
if (process.env.NODE_ENV === 'production') {
  app.use('/api/analysis', auth0Protected, analysisRoutes);
} else {
  app.use('/api/analysis', analysisRoutes);
}

// Auth0 korumalı kullanıcı rotaları
if (process.env.NODE_ENV === 'production') {
  app.use('/api/user', auth0Protected, userRoutes);
} else {
  app.use('/api/user', userRoutes);
}

app.get('/', (req, res) => {
    res.json({ message: 'Welcome to GitGudWeb API!' });
});

// Test endpoint'i
app.get('/api/test', (req, res) => {
    res.json({ message: 'API is working!' });
});

// Auth0 test endpoint'i
app.get('/api/auth-test', auth0Protected, (req, res) => {
    res.json({ 
        message: 'You are authenticated!',
        user: req.user 
    });
});

// 404 handler
app.use((req, res, next) => {
    res.status(404).json({ message: 'Route bulunamadı' });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    
    // Auth0 hatalarını özel olarak işle
    if (err.name === 'UnauthorizedError') {
        return res.status(401).json({ 
            message: 'Kimlik doğrulama hatası', 
            error: 'Geçersiz veya eksik token' 
        });
    }
    
    res.status(500).json({ 
        message: 'Bir şeyler ters gitti!', 
        error: err.message 
    });
});

// Start server
const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
