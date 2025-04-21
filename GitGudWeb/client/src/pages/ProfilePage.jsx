import { useState, useEffect, useRef } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { useAuth } from '../context/AuthContext';
import { useApi } from '../services/api';
import {
  Box,
  Container,
  Typography,
  Paper,
  Avatar,
  CircularProgress,
  Button,
  TextField,
  Grid,
  Snackbar,
  Alert,
  Divider,
  Card,
  CardContent,
  Tab,
  Tabs
} from '@mui/material';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import PersonIcon from '@mui/icons-material/Person';

function ProfilePage() {
  const { user: auth0User, isAuthenticated, isLoading } = useAuth0();
  const { getToken } = useAuth();
  const [dbUser, setDbUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [editMode, setEditMode] = useState(false);
  const api = useApi();
  const dataFetchedRef = useRef(false);

  // Form state
  const [formData, setFormData] = useState({
    username: '',
    name: '',
    bio: '',
    phone: '',
    address: ''
  });

  // MongoDB'deki kullanıcı bilgilerini al
  useEffect(() => {
    let isMounted = true;
    
    const fetchUserData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const userData = await api.getUserProfile();
        console.log('Alınan kullanıcı verileri:', userData);
        
        if (isMounted) {
          setDbUser(userData);
          setFormData({
            username: userData.username || '',
            name: userData.name || '',
            bio: userData.bio || '',
            phone: userData.phone || '',
            address: userData.address || ''
          });
          
          setLoading(false);
        }
      } catch (error) {
        console.error('Kullanıcı bilgileri alınamadı:', error);
        if (isMounted) {
          setError('Kullanıcı bilgileri alınamadı');
          setLoading(false);
        }
      }
    };

    fetchUserData();
    
    return () => {
      isMounted = false;
    };
  }, [isAuthenticated]);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleEditToggle = () => {
    setEditMode(!editMode);
    
    // Eğer düzenleme modundan çıkıyorsak, form verilerini sıfırla
    if (editMode && dbUser) {
      setFormData({
        username: dbUser.username || '',
        name: dbUser.name || '',
        bio: dbUser.bio || '',
        phone: dbUser.phone || '',
        address: dbUser.address || ''
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      setLoading(true);
      
      // API servisini kullanarak profil güncelleme isteği gönder
      const updatedUser = await api.updateUserProfile(formData);
      setDbUser(updatedUser);
      setSuccess('Profil bilgileriniz başarıyla güncellendi');
      setEditMode(false);
    } catch (error) {
      console.error('Profil güncelleme hatası:', error);
      setError(error.message || 'Profil güncellenirken bir hata oluştu');
    } finally {
      setLoading(false);
    }
  };

  const handleCloseAlert = () => {
    setSuccess(null);
    setError(null);
  };

  if (isLoading || loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        Profil Sayfası
      </Typography>

      <Snackbar open={!!success} autoHideDuration={6000} onClose={handleCloseAlert}>
        <Alert onClose={handleCloseAlert} severity="success" sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>

      <Snackbar open={!!error} autoHideDuration={6000} onClose={handleCloseAlert}>
        <Alert onClose={handleCloseAlert} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="profile tabs">
          <Tab label="Profil Bilgileri" />
          <Tab label="Auth0 Bilgileri" />
        </Tabs>
      </Box>

      {tabValue === 0 && (
        <Card elevation={3} sx={{ mb: 4 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h5">
                Kullanıcı Profili
              </Typography>
              <Button 
                variant={editMode ? "contained" : "outlined"} 
                color={editMode ? "secondary" : "primary"}
                startIcon={editMode ? <SaveIcon /> : <EditIcon />}
                onClick={editMode ? handleSubmit : handleEditToggle}
              >
                {editMode ? 'Kaydet' : 'Düzenle'}
              </Button>
            </Box>

            {dbUser ? (
              <Box component="form" onSubmit={handleSubmit}>
                <Grid container spacing={3}>
                  <Grid item xs={12} display="flex" justifyContent="center" mb={2}>
                    <Avatar 
                      src={dbUser.profilePicture || (auth0User?.picture && auth0User.picture.includes('gravatar.com') ? null : auth0User?.picture)} 
                      alt={dbUser.name || auth0User?.name}
                      sx={{ width: 120, height: 120 }}
                    >
                      {!dbUser.profilePicture && !auth0User?.picture && (
                        <PersonIcon sx={{ width: 80, height: 80 }} />
                      )}
                    </Avatar>
                  </Grid>
                  
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Kullanıcı Adı"
                      name="username"
                      value={formData.username}
                      onChange={handleInputChange}
                      disabled={!editMode}
                      required
                    />
                  </Grid>
                  
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Ad Soyad"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      disabled={!editMode}
                    />
                  </Grid>
                  
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="E-posta"
                      name="email"
                      type="email"
                      value={dbUser?.email || ''}
                      disabled={true}
                      helperText="E-posta adresi Auth0 hesabınızdan alınır ve değiştirilemez"
                    />
                  </Grid>
                  
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Hakkımda"
                      name="bio"
                      multiline
                      rows={4}
                      value={formData.bio}
                      onChange={handleInputChange}
                      disabled={!editMode}
                    />
                  </Grid>
                  
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Telefon"
                      name="phone"
                      value={formData.phone}
                      onChange={handleInputChange}
                      disabled={!editMode}
                    />
                  </Grid>
                  
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Adres"
                      name="address"
                      value={formData.address}
                      onChange={handleInputChange}
                      disabled={!editMode}
                    />
                  </Grid>
                </Grid>
                
                {editMode && (
                  <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                    <Button 
                      variant="outlined" 
                      color="secondary" 
                      onClick={handleEditToggle} 
                      sx={{ mr: 2 }}
                    >
                      İptal
                    </Button>
                    <Button 
                      variant="contained" 
                      color="primary" 
                      type="submit"
                      startIcon={<SaveIcon />}
                    >
                      Kaydet
                    </Button>
                  </Box>
                )}
              </Box>
            ) : (
              <Typography>Kullanıcı bilgileri bulunamadı</Typography>
            )}
          </CardContent>
        </Card>
      )}

      {tabValue === 1 && (
        <Card elevation={3} sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h5" gutterBottom>
              Auth0 Kullanıcı Bilgileri
            </Typography>
            
            {isAuthenticated && auth0User ? (
              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Avatar 
                    src={auth0User.picture} 
                    alt={auth0User.name}
                    sx={{ width: 80, height: 80, mr: 2 }}
                  />
                  <Box>
                    <Typography variant="h6">{auth0User.name}</Typography>
                    <Typography variant="body1">{auth0User.email}</Typography>
                    <Typography variant="body2" color="textSecondary">
                      {auth0User.sub}
                    </Typography>
                  </Box>
                </Box>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="subtitle1" gutterBottom>
                  Auth0 Profil Bilgileri:
                </Typography>
                
                <Box sx={{ bgcolor: 'grey.100', p: 2, borderRadius: 1, fontFamily: 'monospace', fontSize: '0.8rem', overflow: 'auto' }}>
                  <pre>{JSON.stringify(auth0User, null, 2)}</pre>
                </Box>
                
                <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
                  Not: Auth0 profil bilgilerinizi Auth0 yönetim panelinden güncelleyebilirsiniz.
                </Typography>
              </Box>
            ) : (
              <Typography>Giriş yapılmadı</Typography>
            )}
          </CardContent>
        </Card>
      )}
    </Container>
  );
}

export default ProfilePage;
