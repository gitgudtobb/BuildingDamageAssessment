import { useState } from 'react';
import { 
  Dialog, 
  DialogContent, 
  Tabs, 
  Tab, 
  Box 
} from '@mui/material';
import MapImageSelector from './MapImageSelector';
import GoogleMapImageSelector from './GoogleMapImageSelector';
import SatelliteIcon from '@mui/icons-material/Satellite';
import MapIcon from '@mui/icons-material/Map';

function ImageSourceSelector({ open, onClose, onImageSelect }) {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="md" 
      fullWidth
      PaperProps={{
        sx: { 
          borderRadius: 2,
          overflow: 'hidden',
          height: '90vh'
        }
      }}
    >
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange} 
          variant="fullWidth"
          sx={{ bgcolor: '#f5f5f5' }}
        >
          <Tab 
            icon={<SatelliteIcon />} 
            label="Earth Engine" 
            iconPosition="start"
            sx={{ py: 2 }}
          />
          <Tab 
            icon={<MapIcon />} 
            label="Google Maps" 
            iconPosition="start"
            sx={{ py: 2 }}
          />
        </Tabs>
      </Box>
      <DialogContent sx={{ p: 0, overflow: 'hidden' }}>
        {activeTab === 0 ? (
          <MapImageSelector onImageSelect={onImageSelect} onClose={onClose} />
        ) : (
          <GoogleMapImageSelector onImageSelect={onImageSelect} onClose={onClose} />
        )}
      </DialogContent>
    </Dialog>
  );
}

export default ImageSourceSelector;
