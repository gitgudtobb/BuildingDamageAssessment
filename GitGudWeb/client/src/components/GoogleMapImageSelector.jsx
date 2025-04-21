import { useState, useRef, useEffect } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import { 
  Box, 
  Button, 
  Typography, 
  Paper, 
  Fade,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  CircularProgress
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import 'leaflet/dist/leaflet.css';

// API URL
const API_BASE_URL = 'http://localhost:5001';

// Harita türleri
const mapTypes = [
  { value: 'roadmap', label: 'Yol Haritası' },
  { value: 'satellite', label: 'Uydu Görüntüsü' },
  { value: 'terrain', label: 'Arazi' },
  { value: 'hybrid', label: 'Hibrit' }
];

// Component to capture the map instance
function MapController({ onMapReady }) {
  const map = useMap();
  
  useEffect(() => {
    if (map) {
      onMapReady(map);
    }
  }, [map, onMapReady]);
  
  return null;
}

export default function GoogleMapImageSelector({ onImageSelect, onClose }) {
  const [map, setMap] = useState(null);
  const [zoom, setZoom] = useState(15);
  const [mapType, setMapType] = useState('satellite');
  const [step, setStep] = useState('select-area');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [imageMetadata, setImageMetadata] = useState(null);
  const [selectedBounds, setSelectedBounds] = useState(null);
  
  // Harita hazır olduğunda çağrılır
  const handleMapReady = (mapInstance) => {
    setMap(mapInstance);
  };
  
  // Zoom değiştiğinde çağrılır
  const handleZoomChange = (event, newValue) => {
    setZoom(newValue);
    if (map) {
      map.setZoom(newValue);
    }
  };
  
  // Harita tipi değiştiğinde çağrılır
  const handleMapTypeChange = (event) => {
    setMapType(event.target.value);
  };
  
  // Geri butonuna tıklandığında çağrılır
  const handleBack = () => {
    setStep('select-area');
    setPreviewImage(null);
    setImageMetadata(null);
  };
  
  // Görüntü oluştur butonuna tıklandığında çağrılır
  const handleSelectArea = async () => {
    if (!map) return;

    setLoading(true);
    setError(null);

    try {
      const bounds = map.getBounds();
      const center = map.getCenter();
      
      console.log("Harita merkezi:", center);
      console.log("Harita sınırları:", bounds);
      
      // Sunucu üzerinden Google Maps görüntüsünü al
      const response = await fetch(`${API_BASE_URL}/api/google-maps/static-image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          center: {
            lat: center.lat,
            lng: center.lng
          },
          zoom: zoom,
          size: {
            width: 640,
            height: 640
          },
          mapType: mapType,
          bounds: {
            north: bounds.getNorth(),
            south: bounds.getSouth(),
            east: bounds.getEast(),
            west: bounds.getWest(),
          }
        }),
        credentials: 'include'
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Görüntü oluşturulurken bir hata oluştu (${response.status})`);
      }
      
      const data = await response.json();
      console.log("API yanıtı:", data);
      
      // Görüntü URL'sini ve metadata'yı ayarla
      setPreviewImage(`${API_BASE_URL}${data.imageUrl}`);
      setImageMetadata(data.metadata);
      setSelectedBounds(bounds);
      setStep('preview-image');
      
    } catch (error) {
      console.error('Error generating preview:', error);
      setError(error.message || 'Görüntü oluşturulurken bir hata oluştu');
    } finally {
      setLoading(false);
    }
  };

  // Görüntüyü seç butonuna tıklandığında çağrılır
  const handleGetImage = () => {
    if (!previewImage || !imageMetadata) {
      setError('Görüntü bilgileri eksik');
      return;
    }

    try {
      console.log("Seçilen görüntü:", previewImage, imageMetadata);
      // Görüntü ve metadata'yı parent bileşene gönder
      onImageSelect(previewImage, imageMetadata);
      onClose();
    } catch (error) {
      console.error('Error selecting image:', error);
      setError(error.message || 'Görüntü seçilirken bir hata oluştu');
    }
  };
  
  return (
    <Paper 
      elevation={3} 
      sx={{ 
        width: '100%', 
        maxWidth: 800, 
        maxHeight: '90vh',
        overflow: 'auto',
        position: 'relative'
      }}
    >
      <Box sx={{ position: 'relative' }}>
        <Fade in={true}>
          <Box>
            <Box sx={{ p: 3, borderBottom: '1px solid #e0e0e0' }}>
              <Typography variant="h5" gutterBottom>
                Google Maps Görüntüsü Seçimi
              </Typography>
              <Box sx={{ display: 'flex', mb: 2 }}>
                <Box 
                  sx={{ 
                    width: 30, 
                    height: 30, 
                    borderRadius: '50%', 
                    bgcolor: step === 'select-area' ? 'primary.main' : 'grey.400',
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mr: 1
                  }}
                >
                  1
                </Box>
                <Typography 
                  variant="body1" 
                  sx={{ 
                    fontWeight: step === 'select-area' ? 'bold' : 'normal',
                    color: step === 'select-area' ? 'text.primary' : 'text.secondary'
                  }}
                >
                  Bölge Seçimi
                </Typography>
                <Box sx={{ mx: 2 }}>—</Box>
                <Box 
                  sx={{ 
                    width: 30, 
                    height: 30, 
                    borderRadius: '50%', 
                    bgcolor: step === 'preview-image' ? 'primary.main' : 'grey.400',
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mr: 1
                  }}
                >
                  2
                </Box>
                <Typography 
                  variant="body1" 
                  sx={{ 
                    fontWeight: step === 'preview-image' ? 'bold' : 'normal',
                    color: step === 'preview-image' ? 'text.primary' : 'text.secondary'
                  }}
                >
                  Görüntü Önizleme
                </Typography>
              </Box>
            </Box>
            
            {step === 'select-area' ? (
              <>
                <Box sx={{ height: 400, width: '100%' }}>
                  <MapContainer 
                    center={[41.0082, 28.9784]} // İstanbul
                    zoom={zoom} 
                    style={{ height: '100%', width: '100%' }}
                  >
                    <TileLayer
                      attribution='&copy; <a href="https://www.google.com/maps">Google Maps</a>'
                      url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                      maxZoom={20}
                      subdomains={['mt0', 'mt1', 'mt2', 'mt3']}
                    />
                    <MapController onMapReady={handleMapReady} />
                  </MapContainer>
                </Box>
                
                <Box sx={{ p: 3 }}>
                  <Typography gutterBottom>Yakınlaştırma Seviyesi: {zoom}</Typography>
                  <Slider
                    value={zoom}
                    onChange={handleZoomChange}
                    min={10}
                    max={20}
                    step={1}
                    marks
                    valueLabelDisplay="auto"
                    sx={{ mb: 3 }}
                  />
                  
                  <FormControl fullWidth sx={{ mb: 3 }}>
                    <InputLabel id="map-type-label">Harita Tipi</InputLabel>
                    <Select
                      labelId="map-type-label"
                      value={mapType}
                      label="Harita Tipi"
                      onChange={handleMapTypeChange}
                    >
                      {mapTypes.map((type) => (
                        <MenuItem key={type.value} value={type.value}>
                          {type.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                    <Button
                      variant="outlined"
                      onClick={onClose}
                      color="secondary"
                    >
                      İptal
                    </Button>
                    <Button
                      variant="contained"
                      onClick={handleSelectArea}
                      disabled={loading || !map}
                      color="primary"
                      endIcon={loading ? <CircularProgress size={20} /> : <PhotoCameraIcon />}
                    >
                      Görüntü Oluştur
                    </Button>
                  </Box>
                </Box>
              </>
            ) : (
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Görüntü Önizleme
                </Typography>
                
                {previewImage && (
                  <Box 
                    sx={{ 
                      width: '100%', 
                      height: '400px', 
                      backgroundImage: `url(${previewImage})`,
                      backgroundSize: 'cover',
                      backgroundPosition: 'center',
                      border: '1px solid #e0e0e0',
                      borderRadius: 1,
                      mb: 3
                    }} 
                  />
                )}
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                  <Button
                    variant="outlined"
                    onClick={handleBack}
                    disabled={loading}
                    startIcon={<ArrowBackIcon />}
                  >
                    Geri
                  </Button>
                  <Button
                    variant="contained"
                    onClick={handleGetImage}
                    disabled={loading || !previewImage}
                    color="primary"
                    endIcon={loading ? <CircularProgress size={20} /> : <CheckCircleIcon />}
                  >
                    Görüntüyü Seç
                  </Button>
                </Box>
              </Box>
            )}
            
            <Box sx={{ p: 3 }}>
              {error && (
                <Card sx={{ mb: 2, bgcolor: '#ffebee', border: '1px solid #ffcdd2' }}>
                  <CardContent>
                    <Typography color="error" variant="body2">
                      {error}
                    </Typography>
                  </CardContent>
                </Card>
              )}
            </Box>
          </Box>
        </Fade>
        
        {loading && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'rgba(255, 255, 255, 0.7)',
              zIndex: 1000,
            }}
          >
            <Box sx={{ textAlign: 'center' }}>
              <CircularProgress size={60} />
              <Typography variant="h6" sx={{ mt: 2 }}>
                {step === 'select-area' ? 'Görüntü Oluşturuluyor...' : 'Görüntü Yükleniyor...'}
              </Typography>
            </Box>
          </Box>
        )}
      </Box>
    </Paper>
  );
}
