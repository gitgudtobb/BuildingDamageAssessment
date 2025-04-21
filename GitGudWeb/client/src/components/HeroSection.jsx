import React from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Button, 
  Grid, 
  Paper,
  useTheme,
  useMediaQuery,
  Fade
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import FactCheckIcon from '@mui/icons-material/FactCheck';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';

const HeroSection = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  return (
    <Box 
      sx={{ 
        position: 'relative',
        overflow: 'hidden',
        py: isMobile ? 6 : 10,
        background: `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`,
        boxShadow: 'inset 0 -10px 20px -10px rgba(0,0,0,0.2)',
      }}
    >
      {/* Dekoratif elementler */}
      <Box sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        opacity: 0.05,
        background: 'url("data:image/svg+xml,%3Csvg width=\'100\' height=\'100\' viewBox=\'0 0 100 100\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cpath d=\'M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z\' fill=\'%23ffffff\' fill-opacity=\'1\' fill-rule=\'evenodd\'/%3E%3C/svg%3E")',
      }} />

      {/* Ana içerik */}
      <Container maxWidth="lg">
        <Grid container spacing={4} alignItems="center">
          <Grid item xs={12} md={6}>
            <Fade in={true} timeout={1000}>
              <Box>
                <Typography 
                  variant="overline" 
                  sx={{ 
                    color: 'rgba(255,255,255,0.9)',
                    fontWeight: 600,
                    letterSpacing: 2,
                    mb: 1,
                    display: 'block'
                  }}
                >
                  YAPAY ZEKA DESTEKLİ
                </Typography>
                
                <Typography 
                  variant="h2" 
                  component="h1" 
                  sx={{ 
                    fontWeight: 800, 
                    color: '#fff',
                    mb: 2,
                    textShadow: '0px 2px 4px rgba(0,0,0,0.2)',
                    fontSize: { xs: '2.5rem', md: '3.5rem' },
                    lineHeight: 1.2,
                    position: 'relative',
                    '&::after': {
                      content: '""',
                      position: 'absolute',
                      left: 0,
                      bottom: -16,
                      width: 80,
                      height: 6,
                      borderRadius: 3,
                      background: theme.palette.secondary.main,
                      boxShadow: '0 4px 10px rgba(0,0,0,0.2)'
                    }
                  }}
                >
                  Deprem Hasar Analizi
                </Typography>
                
                <Typography 
                  variant="h6" 
                  sx={{ 
                    fontWeight: 400, 
                    color: 'rgba(255,255,255,0.85)',
                    mt: 3,
                    mb: 4,
                    maxWidth: 550,
                    lineHeight: 1.6
                  }}
                >
                  Gelişmiş yapay zeka algoritmalarımız ile deprem sonrası hasar tespitini 
                  hızlı, güvenilir ve hassas bir şekilde gerçekleştiriyoruz.
                </Typography>
                
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Button 
                    variant="contained" 
                    size="large"
                    startIcon={<UploadFileIcon />}
                    sx={{ 
                      bgcolor: theme.palette.secondary.main,
                      color: '#fff', 
                      px: 3,
                      py: 1.5,
                      fontWeight: 600,
                      textTransform: 'none',
                      fontSize: '1rem',
                      borderRadius: '8px',
                      boxShadow: '0 4px 14px rgba(0, 0, 0, 0.2)',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        bgcolor: theme.palette.secondary.dark,
                        transform: 'translateY(-3px)',
                        boxShadow: '0 6px 20px rgba(0, 0, 0, 0.25)'
                      }
                    }}
                  >
                    Hemen Analiz Et
                  </Button>
                  
                  <Button 
                    variant="outlined" 
                    size="large"
                    endIcon={<KeyboardArrowRightIcon />}
                    sx={{ 
                      color: '#fff', 
                      borderColor: 'rgba(255,255,255,0.5)',
                      px: 3,
                      py: 1.5,
                      fontWeight: 500,
                      textTransform: 'none',
                      fontSize: '1rem',
                      borderRadius: '8px',
                      borderWidth: '2px',
                      '&:hover': {
                        borderColor: '#fff',
                        backgroundColor: 'rgba(255,255,255,0.1)'
                      }
                    }}
                  >
                    Nasıl Çalışır?
                  </Button>
                </Box>
              </Box>
            </Fade>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Fade in={true} timeout={1500}>
              <Box 
                sx={{ 
                  position: 'relative',
                  mt: { xs: 4, md: 0 },
                  display: 'flex',
                  justifyContent: 'center'
                }}
              >
                {/* Ana görsel */}
                <Paper
                  elevation={10}
                  sx={{
                    borderRadius: 3,
                    overflow: 'hidden',
                    width: '90%',
                    height: isMobile ? 240 : 340,
                    position: 'relative',
                    backgroundImage: 'url("https://images.unsplash.com/photo-1600096194534-95cf5ece04cf")',
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
                    border: '4px solid rgba(255,255,255,0.1)',
                  }}
                >
                  {/* Overlay */}
                  <Box sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0,0,0,0.3)',
                    zIndex: 1
                  }} />
                  
                  {/* Simüle edilmiş analiz UI overlay */}
                  <Box sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    zIndex: 2,
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center'
                  }}>
                    <Box sx={{
                      width: '80%',
                      height: '80%',
                      border: '2px dashed rgba(255,255,255,0.7)',
                      borderRadius: 2,
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                      alignItems: 'center',
                      backgroundColor: 'rgba(0,0,0,0.4)',
                      backdropFilter: 'blur(3px)'
                    }}>
                      <Typography variant="h6" sx={{ color: 'white', mb: 1, fontWeight: 600 }}>
                        %78 Hasar Tespit Edildi
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)', textAlign: 'center', px: 2 }}>
                        Yüksek risk seviyesi. Detaylı analiz için tıklayın.
                      </Typography>
                    </Box>
                  </Box>
                </Paper>
                
                {/* İstatistik kartları */}
                <Box
                  sx={{
                    position: 'absolute',
                    bottom: -30,
                    right: isMobile ? 'auto' : -20,
                    display: 'flex',
                    flexDirection: isMobile ? 'column' : 'row',
                    gap: 2
                  }}
                >
                  <Paper
                    elevation={6}
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      backgroundColor: '#fff',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1.5,
                      width: 200
                    }}
                  >
                    <AutoGraphIcon sx={{ color: theme.palette.primary.main, fontSize: 32 }} />
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Doğruluk Oranı
                      </Typography>
                      <Typography variant="h6" color="text.primary" fontWeight={600}>
                        %95.6
                      </Typography>
                    </Box>
                  </Paper>
                  
                  <Paper
                    elevation={6}
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      backgroundColor: '#fff',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1.5,
                      width: 200
                    }}
                  >
                    <FactCheckIcon sx={{ color: theme.palette.secondary.main, fontSize: 32 }} />
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Analiz Süresi
                      </Typography>
                      <Typography variant="h6" color="text.primary" fontWeight={600}>
                        5.2 saniye
                      </Typography>
                    </Box>
                  </Paper>
                </Box>
              </Box>
            </Fade>
          </Grid>
        </Grid>
        
        {/* Özellikler bölümü */}
        <Grid container spacing={3} sx={{ mt: isMobile ? 8 : 12 }}>
          {[
            {
              title: 'Hızlı Analiz',
              description: 'Yapay zeka algoritmalarımız saniyeler içinde hasar tespiti yapar.',
              iconColor: theme.palette.success.light
            },
            {
              title: 'Hassas Sonuçlar',
              description: 'Yüksek çözünürlüklü görüntülerle hassas hasar tespiti.',
              iconColor: theme.palette.info.light
            },
            {
              title: 'Güvenilir Veri',
              description: 'Akademik araştırmalarla desteklenen analiz modelleri.',
              iconColor: theme.palette.warning.light
            }
          ].map((item, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Fade in={true} timeout={1000 + (index * 500)}>
                <Paper
                  elevation={4}
                  sx={{
                    p: 3,
                    height: '100%',
                    borderRadius: 3,
                    backgroundColor: 'rgba(255,255,255,0.9)',
                    transition: 'transform 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-5px)',
                      boxShadow: '0 10px 20px rgba(0,0,0,0.15)'
                    }
                  }}
                >
                  <Box
                    sx={{
                      width: 50,
                      height: 50,
                      borderRadius: '50%',
                      backgroundColor: item.iconColor + '20',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mb: 2
                    }}
                  >
                    <Box
                      sx={{
                        width: 15,
                        height: 15,
                        borderRadius: '50%',
                        backgroundColor: item.iconColor
                      }}
                    />
                  </Box>
                  
                  <Typography variant="h6" color="text.primary" sx={{ fontWeight: 600, mb: 1 }}>
                    {item.title}
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary">
                    {item.description}
                  </Typography>
                </Paper>
              </Fade>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default HeroSection;