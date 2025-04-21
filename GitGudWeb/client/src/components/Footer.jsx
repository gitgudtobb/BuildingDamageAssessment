import React from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Link, 
  Grid, 
  Divider,
  useTheme,
  Button,
  IconButton,
  useMediaQuery
} from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import TwitterIcon from '@mui/icons-material/Twitter';
import HomeIcon from '@mui/icons-material/Home';
import InfoIcon from '@mui/icons-material/Info';
import ContactMailIcon from '@mui/icons-material/ContactMail';
import EarthquakeIcon from '@mui/icons-material/Landscape'; // Deprem için temsili ikon

const Footer = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const currentYear = new Date().getFullYear();

  return (
    <Box 
      component="footer" 
      sx={{ 
        py: 6, 
        px: 2, 
        mt: 'auto',
        background: `linear-gradient(135deg, ${theme.palette.primary.light}15, ${theme.palette.secondary.light}25)`,
        borderTop: `1px solid ${theme.palette.divider}`,
        boxShadow: '0px -4px 10px rgba(0, 0, 0, 0.05)'
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4} justifyContent="space-between">
          <Grid item xs={12} sm={4}>
            <Box display="flex" alignItems="center" mb={2}>
              <EarthquakeIcon sx={{ fontSize: 28, mr: 1, color: theme.palette.primary.main }} />
              <Typography 
                variant="h5" 
                component="h2" 
                sx={{ 
                  fontWeight: 700, 
                  background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  letterSpacing: '0.5px'
                }}
              >
                Deprem Analiz
              </Typography>
            </Box>
            <Typography 
              variant="body2" 
              color="text.secondary" 
              sx={{ 
                lineHeight: 1.6,
                mb: 2,
                fontWeight: 400
              }}
            >
              Yapay zeka destekli deprem görüntülerinden analiz çıkaran web uygulaması.
              Deprem sonrası hasar tespiti ve analizi için geliştirilmiştir.
            </Typography>

            <Button 
              variant="outlined" 
              color="primary" 
              size="small"
              sx={{ 
                textTransform: 'none', 
                borderRadius: '20px',
                px: 2,
                '&:hover': {
                  background: theme.palette.primary.main,
                  color: 'white'
                }
              }}
              href="/learn-more"
            >
              Daha Fazla Bilgi
            </Button>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <Typography 
              variant="h6" 
              color="text.primary" 
              gutterBottom 
              sx={{ 
                fontWeight: 'bold',
                position: 'relative',
                '&:after': {
                  content: '""',
                  position: 'absolute',
                  bottom: '-8px',
                  left: 0,
                  width: '40px',
                  height: '3px',
                  backgroundColor: theme.palette.primary.main,
                  borderRadius: '2px'
                }
              }}
            >
              Hızlı Erişim
            </Typography>

            <Box sx={{ mt: 3 }}>
              <Box 
                component={Link} 
                href="/" 
                color="inherit" 
                sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  mb: 1.5, 
                  textDecoration: 'none',
                  transition: 'all 0.2s',
                  '&:hover': {
                    color: theme.palette.primary.main,
                    transform: 'translateX(5px)'
                  }
                }}
              >
                <HomeIcon sx={{ fontSize: 18, mr: 1 }} />
                <Typography variant="body2">Ana Sayfa</Typography>
              </Box>
              
              <Box 
                component={Link} 
                href="/about" 
                color="inherit" 
                sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  mb: 1.5, 
                  textDecoration: 'none',
                  transition: 'all 0.2s',
                  '&:hover': {
                    color: theme.palette.primary.main,
                    transform: 'translateX(5px)'
                  }
                }}
              >
                <InfoIcon sx={{ fontSize: 18, mr: 1 }} />
                <Typography variant="body2">Hakkımızda</Typography>
              </Box>
              
              <Box 
                component={Link} 
                href="/contact" 
                color="inherit" 
                sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  mb: 1.5,
                  textDecoration: 'none',
                  transition: 'all 0.2s',
                  '&:hover': {
                    color: theme.palette.primary.main,
                    transform: 'translateX(5px)'
                  }
                }}
              >
                <ContactMailIcon sx={{ fontSize: 18, mr: 1 }} />
                <Typography variant="body2">İletişim</Typography>
              </Box>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <Typography 
              variant="h6" 
              color="text.primary" 
              gutterBottom 
              sx={{ 
                fontWeight: 'bold',
                position: 'relative',
                '&:after': {
                  content: '""',
                  position: 'absolute',
                  bottom: '-8px',
                  left: 0,
                  width: '40px',
                  height: '3px',
                  backgroundColor: theme.palette.primary.main,
                  borderRadius: '2px'
                }
              }}
            >
              Bizi Takip Edin
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 1, mt: 3 }}>
              <IconButton 
                aria-label="github" 
                sx={{ 
                  backgroundColor: theme.palette.grey[200],
                  transition: 'all 0.3s',
                  '&:hover': {
                    backgroundColor: '#24292e',
                    color: 'white',
                    transform: 'translateY(-3px)'
                  }
                }}
                href='https://github.com/orgs/gitgudtobb/repositories'
              >
                <GitHubIcon fontSize="small" />
              </IconButton>
              
              <IconButton 
                aria-label="linkedin" 
                sx={{ 
                  backgroundColor: theme.palette.grey[200],
                  transition: 'all 0.3s',
                  '&:hover': {
                    backgroundColor: '#0077b5',
                    color: 'white',
                    transform: 'translateY(-3px)'
                  }
                }}
                href='https://www.linkedin.com/school/tobb-ekonomi-ve-teknoloji-%C3%BCniversitesi/'
              >
                <LinkedInIcon fontSize="small" />
              </IconButton>
              
            </Box>

        
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 4, opacity: 0.6 }} />
        
        <Box sx={{ 
          display: 'flex', 
          flexDirection: isMobile ? 'column' : 'row',
          justifyContent: 'space-between',
          alignItems: isMobile ? 'center' : 'flex-start',
          gap: 2
        }}>
          <Typography variant="body2" color="text.secondary" align={isMobile ? 'center' : 'left'}>
            © {currentYear} Deprem Analiz. Tüm hakları saklıdır.
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Link href="/privacy" color="text.secondary" sx={{ 
              fontSize: '0.875rem',
              textDecoration: 'none',
              '&:hover': { color: theme.palette.primary.main }
            }}>
              Gizlilik Politikası
            </Link>
            <Link href="/terms" color="text.secondary" sx={{ 
              fontSize: '0.875rem',
              textDecoration: 'none',
              '&:hover': { color: theme.palette.primary.main }
            }}>
              Kullanım Koşulları
            </Link>
          </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;