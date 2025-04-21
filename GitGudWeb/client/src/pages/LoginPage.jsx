import { useAuth } from '../context/AuthContext';
import {
  Box,
  Container,
  Typography,
  Button,
  Paper,
  Avatar,
  CircularProgress,
  Fade,
  Grow,
  Divider
} from '@mui/material';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import GoogleIcon from '@mui/icons-material/Google';
import EmailIcon from '@mui/icons-material/Email';
import { motion } from 'framer-motion';
import FireAnimation from '../components/FireAnimation';
import LightningAnimation from '../components/LightningAnimation';

function LoginPage() {
  const { loginWithAuth0, isLoading } = useAuth();

  // Email-şifre girişi için Auth0
  const handleEmailLogin = () => {
    loginWithAuth0({
      authorizationParams: {
        connection: 'Username-Password-Authentication'
      }
    });
  };

  // Google ile giriş için
  const handleGoogleLogin = () => {
    loginWithAuth0({
      authorizationParams: {
        connection: 'google-oauth2'
      }
    });
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box
      className="login-page"
      sx={{
        height: '100vh',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #333333 100%)',
        padding: 2,
        position: 'relative',
        overflow: 'hidden'
      }}
    >
      <Box sx={{ position: 'absolute', bottom: 0, left: 0, width: '100%', zIndex: 0 }}>
        <FireAnimation />
      </Box>
      <LightningAnimation />
      <Container maxWidth="sm" sx={{ py: 4, position: 'relative', zIndex: 2 }}>
      <Fade in timeout={1000}>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Box
            sx={{
              mb: 4,
              textAlign: 'center',
              position: 'relative',
              display: 'inline-block',
              width: '100%'
            }}
          >
            <Typography 
              variant="h3" 
              component="h1" 
              gutterBottom 
              align="center" 
              sx={{
                fontWeight: 900,
                letterSpacing: '1px',
                background: 'linear-gradient(45deg, #ff4081 10%, #ffeb3b 50%, #ff9100 90%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
                textShadow: '0 2px 10px rgba(255,64,129,0.3)',
                mb: 0.5,
                textTransform: 'uppercase',
                fontSize: { xs: '1.8rem', sm: '2.2rem', md: '2.5rem' },
                position: 'relative',
                '&::after': {
                  content: '""',
                  position: 'absolute',
                  bottom: '-5px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: '80px',
                  height: '3px',
                  background: 'linear-gradient(45deg, #ff4081 30%, #ff9100 90%)',
                  borderRadius: '2px'
                }
              }}
            >
              <span style={{ color: '#fff' }}>GIT</span>
              <span style={{ color: '#ff4081' }}>GUD</span>
            </Typography>
            <Typography
              variant="h5"
              component="h2"
              align="center"
              sx={{
                fontWeight: 500,
                color: '#fff',
                textShadow: '1px 1px 3px rgba(0,0,0,0.7)',
                mt: 1.5,
                opacity: 0.9,
                fontSize: { xs: '1.2rem', sm: '1.4rem', md: '1.5rem' }
              }}
            >
              Afet Hasar Tespit Sistemi
            </Typography>
          </Box>
          
          <Grow in timeout={1500}>
            <Paper
              component={motion.div}
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
              elevation={6}
              sx={{
                p: 4,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                borderRadius: 2,
                width: '100%',
                background: 'rgba(30, 30, 30, 0.8)',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 8px 32px rgba(255, 0, 0, 0.2)',
                border: '1px solid rgba(255, 255, 255, 0.1)'
                
              }}
            >
              <Avatar sx={{ m: 1, bgcolor: 'error.main', width: 56, height: 56 }}>
                <LockOutlinedIcon fontSize="large" />
              </Avatar>
              <Typography component="h1" variant="h5" sx={{ mb: 3, fontWeight: 'bold', color: '#ffffff' }}>
                Giriş Yap
              </Typography>
              
              <Button
                component={motion.button}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleEmailLogin}
                fullWidth
                variant="contained"
                startIcon={<EmailIcon />}
                sx={{
                  mt: 3,
                  py: 1.5,
                  borderRadius: 2,
                  background: 'linear-gradient(45deg, #ff4081 30%, #ff9100 90%)',
                  boxShadow: '0 3px 15px 2px rgba(255, 64, 129, 0.4)',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #ff4081 10%, #ff9100 70%)',
                    transform: 'translateY(-2px)',
                    transition: 'all 0.3s ease'
                  }
                }}
              >
                Email ve Şifre ile Giriş Yap
              </Button>
              
              <Box sx={{ width: '100%', my: 2, display: 'flex', alignItems: 'center' }}>
                <Divider sx={{ flexGrow: 1 }} />
                <Typography variant="body2" sx={{ mx: 2, color: 'rgba(255, 255, 255, 0.7)' }}>
                  VEYA
                </Typography>
                <Divider sx={{ flexGrow: 1 }} />
              </Box>
              
              <Button
                component={motion.button}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleGoogleLogin}
                fullWidth
                variant="outlined"
                startIcon={<GoogleIcon />}
                sx={{
                  py: 1.5,
                  borderRadius: 2,
                  mb: 2,
                  borderColor: '#ffffff',
                  color: '#ffffff',
                  '&:hover': {
                    borderColor: '#ff4081',
                    color: '#ff4081',
                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    transform: 'translateY(-2px)',
                    transition: 'all 0.3s ease'
                  }
                }}
              >
                Google ile Giriş Yap
              </Button>
            </Paper>
          </Grow>
        </Box>
      </Fade>
    </Container>
    </Box>
  );
}

export default LoginPage;
