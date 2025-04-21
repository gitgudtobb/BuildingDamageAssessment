import { useState, useRef } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import { 
  Box, 
  Button, 
  Typography, 
  CircularProgress, 
  Paper, 
  Stepper, 
  Step, 
  StepLabel,
  Card,
  CardContent,
  Fade
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import dayjs from 'dayjs';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import MapIcon from '@mui/icons-material/Map';
import CalendarMonthIcon from '@mui/icons-material/CalendarMonth';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

// Fix for default marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const API_BASE_URL = 'http://localhost:5001';

const center = {
  lat: 34.052235, // LA coordinates
  lng: -118.243683,
};

function MapImageSelector({ onImageSelect, onClose }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const mapRef = useRef(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedBounds, setSelectedBounds] = useState(null);
  const [step, setStep] = useState('select-area'); // 'select-area' veya 'select-date'

  const handleSelectArea = async () => {
    if (!mapRef.current) return;

    setLoading(true);
    setError(null);

    try {
      const map = mapRef.current;
      const bounds = map.getBounds();

      // Tarihleri al
      const response = await fetch(`${API_BASE_URL}/api/earth-engine/dates`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lat: map.getCenter().lat,
          lng: map.getCenter().lng,
          bounds: {
            north: bounds.getNorth(),
            south: bounds.getSouth(),
            east: bounds.getEast(),
            west: bounds.getWest(),
          },
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Tarihler alınırken bir hata oluştu');
      }

      if (!data.dates || data.dates.length === 0) {
        throw new Error('Bu bölge için kullanılabilir görüntü bulunamadı');
      }

      const dates = data.dates.map((date) => dayjs(date));
      setSelectedDate(dates[0]);
      setAvailableDates(dates);
      setSelectedBounds(bounds);
      setStep('select-date');
    } catch (error) {
      console.error('Error fetching dates:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGetImage = async () => {
    if (!selectedDate || !selectedBounds || !mapRef.current) return;

    setLoading(true);
    setError(null);

    try {
      const map = mapRef.current;
      const viewport = {
        width: map.getContainer().clientWidth,
        height: map.getContainer().clientHeight,
      };

      const response = await fetch(`${API_BASE_URL}/api/earth-engine/image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lat: map.getCenter().lat,
          lng: map.getCenter().lng,
          date: selectedDate.format('YYYY-MM-DD'),
          bounds: {
            north: selectedBounds.getNorth(),
            south: selectedBounds.getSouth(),
            east: selectedBounds.getEast(),
            west: selectedBounds.getWest(),
          },
          viewport: viewport,
          zoom: map.getZoom(),
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Görüntü alınırken bir hata oluştu');
      }

      if (!data.imageUrl) {
        throw new Error('Bu tarih için görüntü bulunamadı');
      }

      onImageSelect(data.imageUrl, data.metadata);
      onClose();
    } catch (error) {
      console.error('Error fetching image:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    setStep('select-area');
    setSelectedDate(null);
    setAvailableDates([]);
    setSelectedBounds(null);
  };

  const steps = ['Bölge Seçimi', 'Tarih Seçimi'];
  const activeStep = step === 'select-area' ? 0 : 1;

  return (
    <Paper elevation={3} sx={{ borderRadius: 2, overflow: 'hidden' }}>
      <Box sx={{ p: 3, bgcolor: '#f5f5f5' }}>
        <Typography variant="h5" fontWeight="bold" sx={{ mb: 2 }}>
          Uydu Görüntüsü Seçimi
        </Typography>
        
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label, index) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Box>
      
      <Box sx={{ width: '100%', position: 'relative' }}>
        <Fade in={true} timeout={500}>
          <Box>
            <Box
              sx={{
                height: '400px',
                position: 'relative',
                '& .leaflet-tile': {
                  crossOrigin: 'anonymous',
                },
                border: '1px solid #e0e0e0',
              }}
            >
              <MapContainer
                center={[center.lat, center.lng]}
                zoom={13}
                style={{ height: '100%', width: '100%' }}
                ref={mapRef}
              >
                <TileLayer
                  url="http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                  maxZoom={20}
                  subdomains={['mt0', 'mt1', 'mt2', 'mt3']}
                  attribution="&copy; Google Maps"
                />
              </MapContainer>
            </Box>
            
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

              {step === 'select-date' && (
                <Box sx={{ mb: 3, display: 'flex', justifyContent: 'center' }}>
                  <LocalizationProvider dateAdapter={AdapterDayjs}>
                    <DatePicker
                      label="Tarih Seçin"
                      value={selectedDate}
                      onChange={(newDate) => setSelectedDate(newDate)}
                      disabled={loading}
                      shouldDisableDate={(date) =>
                        !availableDates.some((d) => d.isSame(date, 'day'))
                      }
                      sx={{ width: '100%', maxWidth: 300 }}
                    />
                  </LocalizationProvider>
                </Box>
              )}

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                {step === 'select-date' ? (
                  <>
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
                      disabled={loading || !selectedDate}
                      color="primary"
                      endIcon={loading ? <CircularProgress size={20} /> : <CheckCircleIcon />}
                    >
                      Görüntüyü Seç
                    </Button>
                  </>
                ) : (
                  <>
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
                      disabled={loading}
                      color="primary"
                      endIcon={loading ? <CircularProgress size={20} /> : <MapIcon />}
                    >
                      Bölgeyi Seç
                    </Button>
                  </>
                )}
              </Box>
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
                {step === 'select-area' ? 'Tarihler Yükleniyor...' : 'Görüntü Yükleniyor...'}
              </Typography>
            </Box>
          </Box>
        )}
      </Box>
    </Paper>
  );
}

export default MapImageSelector;