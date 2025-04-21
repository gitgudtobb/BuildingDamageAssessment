const express = require('express');
const router = express.Router();
const { classicAuth } = require('../controllers/authController');

// Klasik kimlik doğrulama rotaları
router.post('/register', classicAuth.register);
router.post('/login', classicAuth.login);

module.exports = router;
