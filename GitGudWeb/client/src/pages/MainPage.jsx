import { useDropzone } from 'react-dropzone';
import DragDropUploader from "../components/DragDropUploader.jsx";
import { useState, useCallback, useEffect } from 'react'
import { 
  Box, 
  Container, 
  Typography, 
  Paper, 
  IconButton,
  Grid,
  Button,
  useTheme,
  useMediaQuery,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Snackbar,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Fade,
  Grow,
  Divider,
  Chip,
  Tooltip,
  Stack,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import DeleteIcon from '@mui/icons-material/Delete'
import MapIcon from '@mui/icons-material/Map'
import AnalyticsIcon from '@mui/icons-material/Analytics'
import TimelineIcon from '@mui/icons-material/Timeline'
import HistoryIcon from '@mui/icons-material/History'
import CompareIcon from '@mui/icons-material/Compare'
import WarningIcon from '@mui/icons-material/Warning'
import InfoIcon from '@mui/icons-material/Info'
import '../App.css'
import ImageSourceSelector from '../components/ImageSourceSelector'
import Header from '../components/Header'
import Footer from '../components/Footer'
import HeroSection from '../components/HeroSection'
import AIAnalysisPanel from '../components/AIAnalysisPanel'
import pixelmatch from 'pixelmatch'
import Pica from 'pica'
import { useAuth0 } from '@auth0/auth0-react'
import { useNavigate } from 'react-router-dom'

const pica = new Pica()

const API_BASE_URL = 'http://localhost:5001'; // Server'ın çalıştığı port

function MainPage() {
  const { logout, getAccessTokenSilently } = useAuth0();
  const navigate = useNavigate();
  const [images, setImages] = useState([null, null])
  const [mapDialogOpen, setMapDialogOpen] = useState(false)
  const [activeImageIndex, setActiveImageIndex] = useState(0)
  const [selectedAnalysis, setSelectedAnalysis] = useState(null)
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' })
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(false);
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'))

    const handleFilesAdded = (acceptedFiles) => {
    const file = acceptedFiles[0]; 
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const newImages = [...images];
        newImages[activeImageIndex] = e.target.result;
        setImages(newImages);
        setAnalysisResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const onDrop = (acceptedFiles, index) => {  
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const newImages = [...images];
        newImages[index] = e.target.result;
        setImages(newImages);
      };
      reader.readAsDataURL(file);
    }
  };
  
  const {                               
    getRootProps: getRootProps1,
    getInputProps: getInputProps1
  } = useDropzone({
    onDrop: (files) => onDrop(files, 0),
    multiple: false,
    accept: 'image/*',
  });
  
  const {                            
    getRootProps: getRootProps2,
    getInputProps: getInputProps2
  } = useDropzone({
    onDrop: (files) => onDrop(files, 1),
    multiple: false,
    accept: 'image/*',
  });
  const handleImageUpload = (index) => (event) => {
    const file = event.target.files[0]
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        setNotification({
          open: true,
          message: 'Dosya boyutu 10MB\'dan küçük olmalıdır',
          severity: 'error'
        })
        return
      }

      const reader = new FileReader()
      reader.onload = (e) => {
        const newImages = [...images]
        newImages[index] = e.target.result
        setImages(newImages)
        setAnalysisResult(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleImageDelete = (index) => () => {
    const newImages = [...images]
    newImages[index] = null
    setImages(newImages)
    setAnalysisResult(null)
  }

  const handleMapSelect = (index) => () => {
    setActiveImageIndex(index)
    setMapDialogOpen(true)
  }

  const handleMapImageSelect = async (imageUrl) => {
    try {
      setLoading(true);
      console.log('Loading satellite image:', imageUrl);

      // Pre-load the image
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      await new Promise((resolve, reject) => {
        img.onload = () => {
          console.log('Image loaded successfully:', {
            width: img.width,
            height: img.height,
            naturalWidth: img.naturalWidth,
            naturalHeight: img.naturalHeight
          });
          resolve();
        };
        img.onerror = (e) => {
          console.error('Image load error:', e);
          reject(new Error('Görüntü yüklenemedi'));
        };
        img.src = imageUrl;
      });

      // Create new images array
      const newImages = [...images];
      newImages[activeImageIndex] = imageUrl;
      setImages(newImages);

      // Show success notification
      setNotification({
        open: true,
        message: 'Uydu görüntüsü başarıyla yüklendi',
        severity: 'success'
      });

      setMapDialogOpen(false);
    } catch (error) {
      console.error('Error loading satellite image:', error);
      setNotification({
        open: true,
        message: 'Uydu görüntüsü yüklenirken bir hata oluştu: ' + error.message,
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const analyzeDamage = async () => {
    if (!images[0] || !images[1]) {
      setNotification({
        open: true,
        message: 'Lütfen her iki fotoğrafı da yükleyin',
        severity: 'error'
      });
      return;
    }

    setAnalyzing(true);
    setAnalysisResult(null);

    try {
      // API'ye gönder - Python AI modeli ile entegre edilmiş endpoint'e istek yap
      const response = await fetch(`${API_BASE_URL}/api/analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          beforeImage: images[0],
          afterImage: images[1],
        }),
        credentials: 'include'
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Analiz kaydedilirken bir hata oluştu');
      }

      const result = await response.json();
      setAnalysisResult(result.results);
      
      // Başarılı bildirim göster
      setNotification({
        open: true,
        message: 'Yapay zeka analizi başarıyla tamamlandı',
        severity: 'success'
      });

      // Analizleri yeniden yükle
      loadAnalyses();
    } catch (error) {
      console.error('Yapay zeka analiz hatası:', error);
      setNotification({
        open: true,
        message: error.message || 'Yapay zeka analizi sırasında bir hata oluştu',
        severity: 'error'
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const loadAnalyses = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/analysis/my`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        },
        credentials: 'include'
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Analizler yüklenirken bir hata oluştu');
      }
      
      const data = await response.json();
      console.log("Yüklenen analizler:", data);
      setAnalyses(data.analyses || []);
    } catch (error) {
      console.error('Analizleri yükleme hatası:', error);
      setNotification({
        open: true,
        message: 'Analizleri yüklerken bir hata oluştu',
        severity: 'error'
      });
      setAnalyses([]);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalysisClick = (analysis) => {
    setSelectedAnalysis(analysis);
  };

  useEffect(() => {
    loadAnalyses();
  }, []);

  const handleCloseNotification = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setNotification({ ...notification, open: false });
  };

  // Helper function to get damage color
  const getDamageColor = (damagePercentage) => {
    if (damagePercentage < 20) return '#4caf50'; // Green
    if (damagePercentage < 50) return '#ff9800'; // Orange
    return '#f44336'; // Red
  };

  // Format date function
  const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return new Date(dateString).toLocaleDateString('tr-TR', options);
  };

  const renderHeader = () => (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
      <Typography variant="h4" component="h1">
        GitGud Web
      </Typography>
      <Box>
        <Button 
          variant="outlined" 
          color="primary" 
          onClick={() => navigate('/profile')}
          sx={{ mr: 2 }}
        >
          Profil
        </Button>
        <Button 
          variant="outlined" 
          color="error" 
          onClick={() => logout({ returnTo: window.location.origin })}
        >
          Çıkış Yap
        </Button>
      </Box>
    </Box>
  )

    return (
      <Box className="main-page" sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Header />
        
        <Container component="main" maxWidth="lg" sx={{ flexGrow: 1, py: 4 }}>
          <AnimatePresence>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
              <Typography 
                variant="h3" 
                component="h1" 
                gutterBottom 
                sx={{ 
                  fontWeight: 700, 
                  textAlign: 'center',
                  mb: 4,
                  background: 'linear-gradient(45deg, #1976d2, #2196f3)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  textShadow: '0px 2px 4px rgba(0,0,0,0.1)'
                }}
              >
                Deprem Hasar Analizi
              </Typography>
              
              <Paper 
                component={motion.div}
                whileHover={{ boxShadow: '0 14px 35px rgba(0, 0, 0, 0.07)' }}
                elevation={3} 
                sx={{ 
                  p: 3, 
                  mb: 4, 
                  borderRadius: 2,
                  background: 'linear-gradient(to right, #f5f7fa, #ffffff)',
                  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.05)',
                  overflow: 'hidden'
                }}
              >
                <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3, color: '#1976d2' }}>
                  <CompareIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Görüntü Karşılaştırması
                </Typography>
                
                <Grid container spacing={3}>
                  {[0, 1].map((index) => (
                    <Grid item xs={12} md={6} key={index}>
                      <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5, delay: index * 0.2 }}
                      >
                        <Card 
                          elevation={2} 
                          sx={{ 
                            height: '100%', 
                            display: 'flex', 
                            flexDirection: 'column',
                            borderRadius: 2,
                            transition: 'transform 0.3s ease, box-shadow 0.3s ease',
                            '&:hover': {
                              transform: 'translateY(-5px)',
                              boxShadow: '0 12px 20px rgba(0, 0, 0, 0.1)'
                            }
                          }}
                        >
                          <CardContent sx={{ flexGrow: 1 }}>
                            <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                              {index === 0 ? 'Deprem Öncesi' : 'Deprem Sonrası'}
                            </Typography>
                            
                            <Box 
                              className="upload-zone"
                              sx={{ 
                                height: 300, 
                                position: 'relative',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                borderRadius: 2,
                                border: '2px dashed',
                                borderColor: images[index] ? 'primary.main' : 'grey.300',
                                backgroundColor: images[index] ? 'rgba(25, 118, 210, 0.04)' : 'grey.50',
                                transition: 'all 0.3s ease',
                                overflow: 'hidden'
                              }}
                            >
                              {images[index] ? (
                                <>
                                  <Box 
                                    component="img" 
                                    src={images[index]} 
                                    alt={`Image ${index + 1}`} 
                                    sx={{ 
                                      width: '100%', 
                                      height: '100%', 
                                      objectFit: 'cover' 
                                    }} 
                                  />
                                  <Box 
                                    sx={{ 
                                      position: 'absolute', 
                                      bottom: 0, 
                                      left: 0, 
                                      right: 0, 
                                      p: 1, 
                                      backgroundColor: 'rgba(0, 0, 0, 0.6)',
                                      display: 'flex',
                                      justifyContent: 'flex-end'
                                    }}
                                  >
                                    <Tooltip title="Görüntüyü Sil">
                                      <IconButton 
                                        size="small" 
                                        onClick={handleImageDelete(index)}
                                        sx={{ color: 'white' }}
                                      >
                                        <DeleteIcon />
                                      </IconButton>
                                    </Tooltip>
                                    <Tooltip title="Haritadan Seç">
                                      <IconButton 
                                        size="small" 
                                        onClick={handleMapSelect(index)}
                                        sx={{ color: 'white' }}
                                      >
                                        <MapIcon />
                                      </IconButton>
                                    </Tooltip>
                                  </Box>
                                </>
                              ) : (
                                <>
                                  <motion.div
                                    animate={{ scale: [1, 1.05, 1] }}
                                    transition={{ repeat: Infinity, duration: 2 }}
                                  >
                                    <CloudUploadIcon sx={{ fontSize: 60, color: 'grey.400', mb: 2 }} />
                                  </motion.div>
                                  <Typography variant="body1" gutterBottom>
                                    Görüntü Yükleyin
                                  </Typography>
                                  <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                                    veya haritadan seçin
                                  </Typography>
                                  {/* Drag & Drop + Dosya Seçme */}
                                 
                                  <Stack direction="row" spacing={2}>
                                  <Box
                                    {...(index === 0 ? getRootProps1() : getRootProps2())}
                                    sx={{
                                      width: '100%',
                                      display: 'inline-block'
                                      }}
                                  >
                                      <input {...(index === 0 ? getInputProps1() : getInputProps2())} />
                                      <Button
                                        variant="contained"
                                        fullWidth
                                        startIcon={<CloudUploadIcon />}
                                        sx={{
                                          px: 4,           
                                          py: 2,           
                                          fontSize: '1rem', 
                                          minWidth: '180px',
                                          borderRadius: 2
                                        }}
                                      >
                                        Dosya Seç
                                      </Button>
                                    </Box>

                                    <Button
                                      variant="outlined"
                                      onClick={handleMapSelect(index)}
                                      startIcon={<MapIcon />}
                                      sx={{
                                        px: 2,
                                        py: 0.3,
                                        fontSize: '1rem',
                                        minWidth: '180px',
                                        borderRadius: 2
                                      }}
                                    >
                                      Haritadan Seç
                                    </Button>
                                  </Stack>
                                </>
                              )}
                            </Box>
                          </CardContent>
                        </Card>
                      </motion.div>
                    </Grid>
                  ))}
                </Grid>
                
                {/* AI Analiz Paneli */}
                <AIAnalysisPanel 
                  preImage={images[0]} 
                  postImage={images[1]}
                  onAnalysisComplete={(results) => {
                    console.log('AI analiz sonuçları:', results);
                    // Burada analiz sonuçlarıyla ilgili işlemler yapılabilir
                    setNotification({
                      open: true,
                      message: `Analiz tamamlandı! ${results.total_buildings} bina tespit edildi.`,
                      severity: 'success'
                    });
                  }}
                />
              </Paper>
              {/* Analysis Result Section */}
              {analysisResult && (
                <Fade in={!!analysisResult}>
                  <Paper 
                    elevation={3} 
                    sx={{ 
                      p: 3, 
                      mt: 4, 
                      borderRadius: 2,
                      border: '1px solid #2196f3',
                      position: 'relative',
                      overflow: 'hidden'
                    }}
                  >
                    <Box sx={{ 
                      position: 'absolute', 
                      top: 0, 
                      right: 0, 
                      bgcolor: 'primary.main', 
                      color: 'white',
                      px: 2,
                      py: 0.5,
                      borderBottomLeftRadius: 8
                    }}>
                      Yapay Zeka Analizi
                    </Box>
                    
                    <Typography variant="h5" component="h3" gutterBottom>
                      Hasar Analiz Sonuçları
                    </Typography>
                    
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={6}>
                        <Box sx={{ mb: 3 }}>
                          <Typography variant="subtitle1" gutterBottom>
                            Hasar Oranı:
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ 
                              width: '100%', 
                              mr: 1, 
                              bgcolor: 'grey.300',
                              borderRadius: 5,
                              height: 10
                            }}>
                              <Box
                                sx={{
                                  width: `${analysisResult.damagePercentage}%`,
                                  height: '100%',
                                  borderRadius: 5,
                                  bgcolor: getDamageColor(analysisResult.damagePercentage)
                                }}
                              />
                            </Box>
                            <Box sx={{ minWidth: 35 }}>
                              <Typography variant="body2" color="textSecondary">
                                {`${analysisResult.damagePercentage}%`}
                              </Typography>
                            </Box>
                          </Box>
                        </Box>
                        
                        <Typography variant="subtitle1" gutterBottom>
                          Hasar Şiddeti:
                        </Typography>
                        <Chip 
                          label={analysisResult.severity} 
                          color={
                            analysisResult.severity === 'Hafif' ? 'success' :
                            analysisResult.severity === 'Orta' ? 'warning' :
                            analysisResult.severity === 'Orta-Ağır' ? 'error' : 'error'
                          }
                          sx={{ mb: 3 }}
                        />
                        
                        <Typography variant="subtitle1" gutterBottom>
                          Yapay Zeka Önerileri:
                        </Typography>
                        <List>
                          {analysisResult.recommendations.map((rec, index) => (
                            <ListItem key={index} sx={{ py: 0.5 }}>
                              <ListItemIcon sx={{ minWidth: 36 }}>
                                <InfoIcon color="primary" fontSize="small" />
                              </ListItemIcon>
                              <ListItemText primary={rec} />
                            </ListItem>
                          ))}
                        </List>
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" gutterBottom>
                          Yapay Zeka Görsel Analizi:
                        </Typography>
                        {analysisResult.processedImages && (
                          <Box sx={{ mt: 2 }}>
                            <img 
                              src={`${API_BASE_URL}/${analysisResult.processedImages.difference.replace(/\\/g, '/')}`} 
                              alt="Hasar Fark Analizi" 
                              style={{ 
                                width: '100%', 
                                borderRadius: 8,
                                border: '1px solid #ddd'
                              }} 
                            />
                            <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 1, textAlign: 'center' }}>
                              Yapay Zeka Tarafından Tespit Edilen Hasar Bölgeleri
                            </Typography>
                          </Box>
                        )}
                      </Grid>
                    </Grid>
                    
                    <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                      <Button 
                        variant="outlined" 
                        color="primary" 
                        startIcon={<TimelineIcon />}
                        onClick={() => {
                          // Save analysis to history or generate report
                        }}
                      >
                        Rapor Oluştur
                      </Button>
                    </Box>
                  </Paper>
                </Fade>
              )}
              <Paper 
              component={motion.div}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.5 }}
              whileHover={{ boxShadow: '0 14px 35px rgba(0, 0, 0, 0.07)' }}
              elevation={3} 
              sx={{ 
                p: 3, 
                borderRadius: 2,
                background: 'linear-gradient(to right, #f5f7fa, #ffffff)',
                boxShadow: '0 10px 30px rgba(0, 0, 0, 0.05)'
              }}
            >
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3, color: '#1976d2' }}>
                <HistoryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Önceki Analizler
              </Typography>
              
              {loading ? (
                <Box sx={{ width: '100%', my: 4, textAlign: 'center' }}>
                  <CircularProgress />
                  <Typography variant="body1" sx={{ mt: 2 }}>
                    Analizler yükleniyor...
                  </Typography>
                </Box>
              ) : analyses.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <Box 
                    sx={{ 
                      textAlign: 'center', 
                      py: 6,
                      bgcolor: 'rgba(0, 0, 0, 0.02)',
                      borderRadius: 2
                    }}
                  >
                    <InfoIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2, opacity: 0.5 }} />
                    <Typography variant="h6" color="text.secondary">
                      Henüz kaydedilmiş analiz bulunmuyor
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Yeni bir analiz yapmak için yukarıdaki görüntüleri yükleyin
                    </Typography>
                  </Box>
                </motion.div>
              ) : (
                <Grid container spacing={3}>
                  {analyses.map((analysis, index) => (
                    <Grid item xs={12} sm={6} md={4} key={index}>
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1, duration: 0.5 }}
                        whileHover={{ y: -10 }}
                      >
                        <Card 
                          sx={{ 
                            cursor: 'pointer',
                            height: '100%',
                            borderRadius: 2,
                            transition: 'transform 0.3s ease, box-shadow 0.3s ease',
                            '&:hover': {
                              boxShadow: '0 8px 16px rgba(0, 0, 0, 0.1)'
                            }
                          }}
                          onClick={() => handleAnalysisClick(analysis)}
                        >
                          <Box sx={{ position: 'relative', paddingTop: '56.25%' }}>
                            <Box
                              component="img"
                              src={analysis.afterImageUrl}
                              alt="Analiz Sonrası Görüntü"
                              sx={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                              }}
                            />
                            <Box
                              sx={{
                                position: 'absolute',
                                top: 10,
                                right: 10,
                                bgcolor: getDamageColor(analysis.results.damagePercentage),
                                color: 'white',
                                borderRadius: '50%',
                                width: 40,
                                height: 40,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontWeight: 'bold',
                                boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                              }} 
                              >
                              {`${Math.round(analysis.results.damagePercentage)}%`}
                            </Box>
                          </Box>
                          <CardContent>
                            <Typography variant="subtitle1" sx={{ fontWeight: 'medium', mb: 1 }}>
                              {analysis.location?.address || 'Konum bilgisi yok'}
                            </Typography>
                            <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
                              {formatDate(analysis.createdAt || new Date())}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                              <Chip 
                                size="small" 
                                label={
                                  analysis.results.damagePercentage < 20
                                    ? 'Hafif Hasar'
                                    : analysis.results.damagePercentage < 50
                                    ? 'Orta Hasar'
                                    : 'Ağır Hasar'
                                }
                                sx={{ 
                                  bgcolor: getDamageColor(analysis.results.damagePercentage) + '20',
                                  color: getDamageColor(analysis.results.damagePercentage),
                                  fontWeight: 'medium'
                                }}
                              />
                              <Chip 
                                size="small" 
                                label={analysis.metadata?.buildingType || 'Bina'}
                                variant="outlined"
                              />
                            </Box>
                          </CardContent>
                        </Card>
                      </motion.div>
                    </Grid>
                  ))}
                </Grid>
              )}
            </Paper>
          </motion.div>
        </AnimatePresence>
      </Container>
      <Footer />
      
      {/* Image Source Selector Dialog */}
      <Dialog
        open={mapDialogOpen}
        onClose={() => setMapDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <ImageSourceSelector 
          open={mapDialogOpen} 
          onClose={() => setMapDialogOpen(false)} 
          onImageSelect={handleMapImageSelect} 
        />
      </Dialog>
      
      {/* Analysis Detail Dialog */}
      <Dialog
        open={!!selectedAnalysis}
        onClose={() => setSelectedAnalysis(null)}
        maxWidth="md"
        fullWidth
      >
        {selectedAnalysis && (
          <>
            <DialogTitle>
              Analiz Detayları
              <IconButton
                aria-label="close"
                onClick={() => setSelectedAnalysis(null)}
                sx={{
                  position: 'absolute',
                  right: 8,
                  top: 8,
                  color: (theme) => theme.palette.grey[500],
                }}
              >
                <DeleteIcon />
              </IconButton>
            </DialogTitle>
            <DialogContent dividers>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Görüntü Karşılaştırması
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <Box sx={{ position: 'relative', paddingTop: '56.25%', mb: 1 }}>
                        <Box
                          component="img"
                          src={selectedAnalysis.beforeImageUrl}
                          alt="Deprem Öncesi"
                          sx={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            borderRadius: 1
                          }}
                        />
                        <Box
                          sx={{
                            position: 'absolute',
                            bottom: 0,
                            left: 0,
                            right: 0,
                            bgcolor: 'rgba(0, 0, 0, 0.6)',
                            color: 'white',
                            p: 1
                          }}
                        >
                          Deprem Öncesi
                        </Box>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Box sx={{ position: 'relative', paddingTop: '56.25%', mb: 1 }}>
                        <Box
                          component="img"
                          src={selectedAnalysis.afterImageUrl}
                          alt="Deprem Sonrası"
                          sx={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            borderRadius: 1
                          }}
                        />
                        <Box
                          sx={{
                            position: 'absolute',
                            bottom: 0,
                            left: 0,
                            right: 0,
                            bgcolor: 'rgba(0, 0, 0, 0.6)',
                            color: 'white',
                            p: 1
                          }}
                        >
                          Deprem Sonrası
                        </Box>
                      </Box>
                    </Grid>
                  </Grid>
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <Card sx={{ height: '100%', borderRadius: 2 }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Hasar Oranı
                      </Typography>
                      <Box sx={{ position: 'relative', display: 'inline-flex', width: '100%', justifyContent: 'center', my: 2 }}>
                        <Box
                          sx={{
                            position: 'relative',
                            display: 'inline-flex',
                            width: 150,
                            height: 150,
                          }}
                        >
                          <CircularProgress
                            variant="determinate"
                            value={100}
                            size={150}
                            thickness={4}
                            sx={{ color: theme.palette.grey[200], position: 'absolute' }}
                          />
                          <CircularProgress
                            variant="determinate"
                            value={selectedAnalysis.results.damagePercentage}
                            size={150}
                            thickness={4}
                            sx={{ color: getDamageColor(selectedAnalysis.results.damagePercentage) }}
                          />
                          <Box
                            sx={{
                              top: 0,
                              left: 0,
                              bottom: 0,
                              right: 0,
                              position: 'absolute',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            <Typography
                              variant="h4"
                              component="div"
                              color={getDamageColor(selectedAnalysis.results.damagePercentage)}
                              fontWeight="bold"
                            >
                              {`${Math.round(selectedAnalysis.results.damagePercentage)}%`}
                            </Typography>
                          </Box>
                        </Box>
                      </Box>
                      <Typography variant="body1" align="center" sx={{ mt: 2 }}>
                        {selectedAnalysis.results.damagePercentage < 20
                          ? 'Hafif Hasar'
                          : selectedAnalysis.results.damagePercentage < 50
                          ? 'Orta Hasar'
                          : 'Ağır Hasar'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <Card sx={{ height: '100%', borderRadius: 2 }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Bina Bilgileri
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        <Grid container spacing={2}>
                          <Grid item xs={6}>
                            <Typography variant="subtitle2" color="textSecondary">
                              Bina Tipi
                            </Typography>
                            <Typography variant="body1" sx={{ mb: 2 }}>
                              {selectedAnalysis.metadata?.buildingType || 'Belirtilmemiş'}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="subtitle2" color="textSecondary">
                              Yapım Yılı
                            </Typography>
                            <Typography variant="body1" sx={{ mb: 2 }}>
                              {selectedAnalysis.metadata?.constructionYear || 'Belirtilmemiş'}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="subtitle2" color="textSecondary">
                              Kat Sayısı
                            </Typography>
                            <Typography variant="body1" sx={{ mb: 2 }}>
                              {selectedAnalysis.metadata?.floorCount || 'Belirtilmemiş'}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="subtitle2" color="textSecondary">
                              Analiz Tarihi
                            </Typography>
                            <Typography variant="body1" sx={{ mb: 2 }}>
                              {formatDate(selectedAnalysis.createdAt || new Date())}
                            </Typography>
                          </Grid>
                        </Grid>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12}>
                  <Card sx={{ borderRadius: 2 }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Değerlendirme ve Öneriler
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        {selectedAnalysis.results.recommendations.map((rec, idx) => (
                          <Box 
                            key={idx}
                            sx={{ 
                              mb: 2, 
                              p: 2, 
                              borderRadius: 1, 
                              bgcolor: 'rgba(25, 118, 210, 0.04)',
                              border: '1px solid rgba(25, 118, 210, 0.1)'
                            }}
                          >
                            <Typography variant="body1">
                              {rec}
                            </Typography>
                          </Box>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setSelectedAnalysis(null)}>Kapat</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
      
      {/* Notification Snackbar */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        sx={{ '& .MuiSnackbarContent-root': { width: '100%' } }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity} 
          sx={{ width: '100%' }}
          variant="filled"
          elevation={6}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  )
}

export default MainPage
