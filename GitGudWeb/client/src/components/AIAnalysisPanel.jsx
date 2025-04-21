import React, { useState } from 'react';
import {
  Box,
  Button,
  Typography,
  CircularProgress,
  Paper,
  Grid,
  Divider,
  Chip,
  Tooltip,
  Alert,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import WarningIcon from '@mui/icons-material/Warning';
import DangerousIcon from '@mui/icons-material/Dangerous';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { useApi } from '../services/api';
import { fixBase64Padding, resizeImage } from '../utils/imageUtils';

const AIAnalysisPanel = ({ preImage, postImage, onAnalysisComplete }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const api = useApi();

  // Hasar seviyelerine göre renk ve ikon atamaları
  const damageConfig = {
    'no-damage': {
      color: '#4caf50',
      icon: <CheckCircleIcon />,
      label: 'Hasar Yok'
    },
    'minor-damage': {
      color: '#ff9800',
      icon: <WarningIcon />,
      label: 'Küçük Hasar'
    },
    'major-damage': {
      color: '#f44336',
      icon: <ErrorIcon />,
      label: 'Büyük Hasar'
    },
    'destroyed': {
      color: '#9c27b0',
      icon: <DangerousIcon />,
      label: 'Yıkılmış'
    }
  };

  const runAnalysis = async () => {
    if (!preImage || !postImage) {
      setError('Analiz için afet öncesi ve sonrası görüntüler gereklidir');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      // Görüntüleri yeniden boyutlandır ve base64 formatını düzelt
      const resizedPreImage = await resizeImage(preImage, 800, 600);
      const resizedPostImage = await resizeImage(postImage, 800, 600);
      
      // Base64 formatını düzelt
      const fixedPreImage = fixBase64Padding(resizedPreImage);
      const fixedPostImage = fixBase64Padding(resizedPostImage);
      
      console.log('Görüntüler düzeltildi ve yeniden boyutlandırıldı');
      
      // AI analiz API'sine istek gönder
      const response = await api.analyzeDamage({
        preImage: fixedPreImage,
        postImage: fixedPostImage
      });

      setResults(response);
      
      // Ebeveyn bileşene sonuçları bildir
      if (onAnalysisComplete) {
        onAnalysisComplete(response);
      }
    } catch (err) {
      console.error('AI analiz hatası:', err);
      setError(err.message || 'Analiz sırasında bir hata oluştu');
    } finally {
      setLoading(false);
    }
  };

  // İstatistik kartı
  const StatCard = ({ title, value, color, icon }) => (
    <Card sx={{ height: '100%', bgcolor: color, color: 'white' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Typography variant="h6">{title}</Typography>
          {icon}
        </Box>
        <Typography variant="h4" fontWeight="bold" mt={1}>
          {value}
        </Typography>
      </CardContent>
    </Card>
  );

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
      <Box display="flex" alignItems="center" mb={2}>
        <AnalyticsIcon fontSize="large" color="primary" sx={{ mr: 1 }} />
        <Typography variant="h5" fontWeight="bold">
          Yapay Zeka Hasar Analizi
        </Typography>
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      {!results && (
        <Box textAlign="center" py={2}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <AnalyticsIcon />}
            onClick={runAnalysis}
            disabled={loading || !preImage || !postImage}
          >
            {loading ? 'Analiz Yapılıyor...' : 'Hasar Analizi Başlat'}
          </Button>
          
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
          
          {(!preImage || !postImage) && (
            <Alert severity="info" sx={{ mt: 2 }}>
              Analiz için afet öncesi ve sonrası görüntüleri seçmelisiniz.
            </Alert>
          )}
        </Box>
      )}
      
      {results && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Analiz Sonuçları
          </Typography>
          
          <Grid container spacing={2} mb={3}>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Toplam Bina"
                value={results.total_buildings}
                color="#2196f3"
                icon={<AnalyticsIcon />}
              />
            </Grid>
            
            {Object.entries(results.statistics).map(([damage, count]) => (
              <Grid item xs={12} sm={6} md={3} key={damage}>
                <StatCard
                  title={damageConfig[damage].label}
                  value={count}
                  color={damageConfig[damage].color}
                  icon={damageConfig[damage].icon}
                />
              </Grid>
            ))}
          </Grid>
          
          {/* Maskelenmiş Görüntü */}
          {results.masked_image && (
            <Box mb={3}>
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                Hasar Analizi Görselleştirmesi:
              </Typography>
              <Paper elevation={2} sx={{ p: 1, textAlign: 'center' }}>
                <img 
                  src={results.masked_image} 
                  alt="Hasar Analizi" 
                  style={{ maxWidth: '100%', maxHeight: '400px' }}
                />
                <Typography variant="caption" display="block" mt={1} color="text.secondary">
                  Yeşil: Hasarsız | Sarı: Az Hasarlı | Turuncu: Çok Hasarlı | Kırmızı: Yıkılmış
                </Typography>
              </Paper>
            </Box>
          )}
          
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Bina Detayları:
          </Typography>
          
          <List>
            {results.buildings.map((building, index) => (
              <ListItem key={index} divider>
                <ListItemIcon>
                  {damageConfig[building.damage].icon}
                </ListItemIcon>
                <ListItemText
                  primary={`Bina #${index + 1}: ${damageConfig[building.damage].label}`}
                  secondary={`Koordinatlar: [${building.bbox.join(', ')}]`}
                />
                <Chip
                  label={damageConfig[building.damage].label}
                  sx={{
                    bgcolor: damageConfig[building.damage].color,
                    color: 'white'
                  }}
                />
              </ListItem>
            ))}
          </List>
          
          <Box mt={3} textAlign="center">
            <Button
              variant="outlined"
              color="primary"
              onClick={() => setResults(null)}
            >
              Yeni Analiz
            </Button>
          </Box>
        </Box>
      )}
    </Paper>
  );
};

export default AIAnalysisPanel;
