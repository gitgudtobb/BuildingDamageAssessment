import { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Alert, 
  CircularProgress,
  Divider,
  Link
} from '@mui/material';
import { motion } from 'framer-motion';
import { useAuth } from '../context/AuthContext';

const ClassicLoginForm = ({ onToggleForm }) => {
  const [isRegister, setIsRegister] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [errors, setErrors] = useState({});
  const [apiError, setApiError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { classicLogin, classicRegister } = useAuth();

  const validateForm = () => {
    const newErrors = {};
    
    if (isRegister && !formData.username.trim()) {
      newErrors.username = 'Kullanıcı adı gerekli';
    }
    
    if (!formData.email.trim()) {
      newErrors.email = 'Email gerekli';
    } else if (!/^\S+@\S+\.\S+$/.test(formData.email)) {
      newErrors.email = 'Geçerli bir email adresi girin';
    }
    
    if (!formData.password) {
      newErrors.password = 'Şifre gerekli';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Şifre en az 6 karakter olmalı';
    }
    
    if (isRegister && formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Şifreler eşleşmiyor';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    
    // Clear error when user types
    if (errors[name]) {
      setErrors({
        ...errors,
        [name]: ''
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setApiError('');
    
    if (!validateForm()) {
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      if (isRegister) {
        // Register
        await classicRegister(
          formData.username,
          formData.email,
          formData.password
        );
      } else {
        // Login
        await classicLogin(formData.email, formData.password);
      }
    } catch (error) {
      setApiError(error.message || 'Bir hata oluştu. Lütfen tekrar deneyin.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const toggleForm = () => {
    setIsRegister(!isRegister);
    setApiError('');
    setErrors({});
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
      {apiError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {apiError}
        </Alert>
      )}
      
      {isRegister && (
        <TextField
          margin="normal"
          required
          fullWidth
          id="username"
          label="Kullanıcı Adı"
          name="username"
          autoComplete="username"
          value={formData.username}
          onChange={handleChange}
          error={!!errors.username}
          helperText={errors.username}
          disabled={isSubmitting}
        />
      )}
      
      <TextField
        margin="normal"
        required
        fullWidth
        id="email"
        label="Email Adresi"
        name="email"
        autoComplete="email"
        value={formData.email}
        onChange={handleChange}
        error={!!errors.email}
        helperText={errors.email}
        disabled={isSubmitting}
      />
      
      <TextField
        margin="normal"
        required
        fullWidth
        name="password"
        label="Şifre"
        type="password"
        id="password"
        autoComplete={isRegister ? 'new-password' : 'current-password'}
        value={formData.password}
        onChange={handleChange}
        error={!!errors.password}
        helperText={errors.password}
        disabled={isSubmitting}
      />
      
      {isRegister && (
        <TextField
          margin="normal"
          required
          fullWidth
          name="confirmPassword"
          label="Şifreyi Onayla"
          type="password"
          id="confirmPassword"
          autoComplete="new-password"
          value={formData.confirmPassword}
          onChange={handleChange}
          error={!!errors.confirmPassword}
          helperText={errors.confirmPassword}
          disabled={isSubmitting}
        />
      )}
      
      <Button
        component={motion.button}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        type="submit"
        fullWidth
        variant="contained"
        sx={{
          mt: 3,
          mb: 2,
          py: 1.5,
          borderRadius: 2,
          background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
          boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
        }}
        disabled={isSubmitting}
      >
        {isSubmitting ? (
          <CircularProgress size={24} color="inherit" />
        ) : (
          isRegister ? 'Kayıt Ol' : 'Giriş Yap'
        )}
      </Button>
      
      <Box sx={{ textAlign: 'center', mt: 2 }}>
        <Typography variant="body2">
          {isRegister ? 'Zaten hesabınız var mı?' : 'Hesabınız yok mu?'}
          {' '}
          <Link
            component="button"
            variant="body2"
            onClick={toggleForm}
            sx={{ fontWeight: 'bold' }}
          >
            {isRegister ? 'Giriş Yap' : 'Kayıt Ol'}
          </Link>
        </Typography>
      </Box>
      
      <Divider sx={{ my: 3 }} />
      
      <Box sx={{ textAlign: 'center' }}>
        <Typography variant="body2" sx={{ mb: 1 }}>
          Veya farklı bir yöntemle giriş yapmak için:
        </Typography>
        <Button
          variant="text"
          color="primary"
          onClick={onToggleForm}
          sx={{ textTransform: 'none' }}
        >
          Auth0 ile giriş yap
        </Button>
      </Box>
    </Box>
  );
};

export default ClassicLoginForm;
