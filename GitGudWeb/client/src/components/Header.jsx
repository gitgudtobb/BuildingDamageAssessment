import React, { useState } from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  Box, 
  Container,
  useTheme,
  useMediaQuery,
  IconButton,
  Menu,
  MenuItem,
  Divider,
  Avatar,
  Tooltip,
  Fade
} from '@mui/material';
import { useAuth0 } from '@auth0/auth0-react';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import HomeIcon from '@mui/icons-material/Home';
import InfoIcon from '@mui/icons-material/Info';
import ContactMailIcon from '@mui/icons-material/ContactMail';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import LogoutIcon from '@mui/icons-material/Logout';
import MenuIcon from '@mui/icons-material/Menu';

const Header = () => {
  const { logout, isAuthenticated, loginWithRedirect, user } = useAuth0();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // Menü durumları
  const [mobileMenuAnchor, setMobileMenuAnchor] = useState(null);
  const [userMenuAnchor, setUserMenuAnchor] = useState(null);
  
  // Mobil menü işleyicileri
  const handleMobileMenuOpen = (event) => setMobileMenuAnchor(event.currentTarget);
  const handleMobileMenuClose = () => setMobileMenuAnchor(null);
  
  // Kullanıcı menü işleyicileri
  const handleUserMenuOpen = (event) => setUserMenuAnchor(event.currentTarget);
  const handleUserMenuClose = () => setUserMenuAnchor(null);
  
  // Navigasyon öğeleri
  const navItems = [
    // { name: 'Ana Sayfa', icon: <HomeIcon fontSize="small" />, path: '/' },
    // { name: 'Hakkımızda', icon: <InfoIcon fontSize="small" />, path: '/about' },
    // { name: 'İletişim', icon: <ContactMailIcon fontSize="small" />, path: '/contact' },
  ];

  return (
    <AppBar 
      position="sticky" 
      elevation={0}
      sx={{ 
        background: `linear-gradient(90deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 70%, ${theme.palette.primary.light} 100%)`,
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(8px)',
      }}
    >
      <Container maxWidth="lg">
        <Toolbar disableGutters sx={{ py: 1 }}>
          {/* Logo ve İsim - Masaüstü */}
          <Box sx={{ 
            display: { xs: 'none', md: 'flex' }, 
            alignItems: 'center',
            mr: 4
          }}>
            <WarningAmberIcon sx={{ 
              mr: 1.5, 
              fontSize: '2.2rem',
              color: '#fff',
              filter: 'drop-shadow(0 0 2px rgba(255,255,255,0.3))'
            }} />
            <Typography
              variant="h5"
              component="a"
              href="/"
              sx={{
                fontFamily: 'Poppins, sans-serif',
                fontWeight: 800,
                color: 'white',
                textDecoration: 'none',
                letterSpacing: '1px',
                textShadow: '0 2px 10px rgba(0,0,0,0.2)',
                position: 'relative',
                '&:after': {
                  content: '""',
                  position: 'absolute',
                  bottom: -8,
                  left: 0,
                  width: '100%',
                  height: '3px',
                  backgroundColor: 'rgba(255,255,255,0.6)',
                  transform: 'scaleX(0)',
                  transition: 'transform 0.3s ease',
                  transformOrigin: 'left'
                },
                '&:hover:after': {
                  transform: 'scaleX(1)'
                }
              }}
            >
              DEPREM ANALİZ
            </Typography>
          </Box>

          {/* Logo ve İsim - Mobil */}
          <Box sx={{ 
            display: { xs: 'flex', md: 'none' }, 
            alignItems: 'center',
            mr: 2
          }}>
            <WarningAmberIcon sx={{ 
              mr: 1, 
              fontSize: '1.8rem',
              color: '#fff' 
            }} />
            <Typography
              variant="h6"
              component="a"
              href="/"
              sx={{
                fontFamily: 'Poppins, sans-serif',
                fontWeight: 700,
                color: 'white',
                textDecoration: 'none',
                letterSpacing: '0.5px',
              }}
            >
              DEPREM ANALİZ
            </Typography>
          </Box>

          {/* Mobil Menü Butonu */}
          <Box sx={{ display: { xs: 'flex', md: 'none' }, ml: 'auto', mr: 1 }}>
            <IconButton
              size="large"
              color="inherit"
              aria-label="open menu"
              onClick={handleMobileMenuOpen}
              sx={{
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)'
                }
              }}
            >
              <MenuIcon />
            </IconButton>
            <Menu
              anchorEl={mobileMenuAnchor}
              open={Boolean(mobileMenuAnchor)}
              onClose={handleMobileMenuClose}
              TransitionComponent={Fade}
              PaperProps={{
                sx: {
                  mt: 1.5,
                  width: 200,
                  boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
                }
              }}
            >
              {navItems.map((item) => (
                <MenuItem 
                  key={item.name} 
                  onClick={() => {
                    handleMobileMenuClose();
                    window.location.href = item.path;
                  }}
                  sx={{
                    py: 1.5,
                    '&:hover': {
                      backgroundColor: theme.palette.primary.light + '20'
                    }
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {item.icon}
                    <Typography sx={{ ml: 1.5 }}>{item.name}</Typography>
                  </Box>
                </MenuItem>
              ))}
              <Divider />
              {isAuthenticated ? (
                [
                  <MenuItem 
                    key="profile"
                    onClick={() => {
                      handleMobileMenuClose();
                      window.location.href = '/profile';
                    }}
                    sx={{
                      py: 1.5,
                      '&:hover': {
                        backgroundColor: theme.palette.primary.light + '20'
                      }
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <AccountCircleIcon fontSize="small" />
                      <Typography sx={{ ml: 1.5 }}>Profil</Typography>
                    </Box>
                  </MenuItem>,
                  <MenuItem 
                    key="logout"
                    onClick={() => {
                      handleMobileMenuClose();
                      logout({ returnTo: window.location.origin });
                    }}
                    sx={{
                      py: 1.5,
                      '&:hover': {
                        backgroundColor: theme.palette.primary.light + '20'
                      }
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <LogoutIcon fontSize="small" />
                      <Typography sx={{ ml: 1.5 }}>Çıkış Yap</Typography>
                    </Box>
                  </MenuItem>
                ]
              ) : (
                <MenuItem 
                  onClick={() => {
                    handleMobileMenuClose();
                    loginWithRedirect();
                  }}
                  sx={{
                    py: 1.5,
                    '&:hover': {
                      backgroundColor: theme.palette.primary.light + '20'
                    }
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <AccountCircleIcon fontSize="small" />
                    <Typography sx={{ ml: 1.5 }}>Giriş Yap</Typography>
                  </Box>
                </MenuItem>
              )}
            </Menu>
          </Box>
          
          {/* Masaüstü Navigasyon */}
          <Box 
            sx={{ 
              flexGrow: 1, 
              display: { xs: 'none', md: 'flex' },
              justifyContent: 'center',
              gap: 2
            }}
          >
            {navItems.map((item) => (
              <Button
                key={item.name}
                href={item.path}
                startIcon={item.icon}
                sx={{
                  color: 'white',
                  py: 1,
                  px: 2,
                  borderRadius: '4px',
                  textTransform: 'none',
                  fontWeight: 500,
                  fontSize: '0.95rem',
                  position: 'relative',
                  overflow: 'hidden',
                  transition: 'all 0.3s ease',
                  '&:before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    transform: 'scaleX(0)',
                    transformOrigin: 'right',
                    transition: 'transform 0.3s ease',
                    zIndex: -1
                  },
                  '&:hover': {
                    backgroundColor: 'transparent',
                    transform: 'translateY(-2px)',
                    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)'
                  },
                  '&:hover:before': {
                    transform: 'scaleX(1)',
                    transformOrigin: 'left'
                  }
                }}
              >
                {item.name}
              </Button>
            ))}
          </Box>
          
          {/* Kullanıcı Profil Bölümü - Masaüstü */}
          <Box sx={{ display: { xs: 'none', md: 'flex' }, ml: 'auto', alignItems: 'center' }}>
            {isAuthenticated ? (
              <>
                <Tooltip title="Kullanıcı ayarları">
                  <Button
                    onClick={handleUserMenuOpen}
                    variant="contained"
                    endIcon={<KeyboardArrowDownIcon />}
                    sx={{
                      backgroundColor: 'rgba(255, 255, 255, 0.15)',
                      color: 'white',
                      textTransform: 'none',
                      px: 2,
                      py: 0.8,
                      borderRadius: '20px',
                      '&:hover': {
                        backgroundColor: 'rgba(255, 255, 255, 0.25)',
                      }
                    }}
                  >
                    <Avatar 
                      src={user?.picture} 
                      alt={user?.name}
                      sx={{ 
                        width: 28, 
                        height: 28, 
                        mr: 1,
                        border: '1px solid rgba(255,255,255,0.5)'
                      }}
                    />
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {user?.name?.split(' ')[0] || 'Kullanıcı'}
                    </Typography>
                  </Button>
                </Tooltip>
                <Menu
                  anchorEl={userMenuAnchor}
                  open={Boolean(userMenuAnchor)}
                  onClose={handleUserMenuClose}
                  TransitionComponent={Fade}
                  PaperProps={{
                    sx: {
                      mt: 1.5,
                      minWidth: 180,
                      boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
                    }
                  }}
                >
                  <MenuItem 
                    onClick={() => {
                      handleUserMenuClose();
                      window.location.href = '/profile';
                    }}
                    sx={{
                      py: 1.5,
                      '&:hover': {
                        backgroundColor: theme.palette.primary.light + '20'
                      }
                    }}
                  >
                    <AccountCircleIcon fontSize="small" sx={{ mr: 1.5 }} />
                    Profil
                  </MenuItem>
                  <Divider />
                  <MenuItem 
                    onClick={() => {
                      handleUserMenuClose();
                      logout({ returnTo: window.location.origin });
                    }}
                    sx={{
                      py: 1.5,
                      '&:hover': {
                        backgroundColor: theme.palette.primary.light + '20'
                      }
                    }}
                  >
                    <LogoutIcon fontSize="small" sx={{ mr: 1.5 }} />
                    Çıkış Yap
                  </MenuItem>
                </Menu>
              </>
            ) : (
              <Button 
                variant="contained" 
                onClick={() => loginWithRedirect()}
                startIcon={<AccountCircleIcon />}
                sx={{ 
                  backgroundColor: 'rgba(255, 255, 255, 0.15)',
                  color: 'white',
                  textTransform: 'none',
                  borderRadius: '20px',
                  px: 2,
                  fontWeight: 500,
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.25)',
                    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)'
                  }
                }}
              >
                Giriş Yap
              </Button>
            )}
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Header;