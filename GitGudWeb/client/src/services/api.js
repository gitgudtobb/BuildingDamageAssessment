import { useAuth } from '../context/AuthContext';
import { useState, useCallback } from 'react';

// API temel URL'i
const API_URL = 'http://localhost:5001/api';

// API istekleri için hook
export const useApi = () => {
  const { getToken, isAuthenticated } = useAuth();
  const [cache, setCache] = useState({});
  const [pendingRequests, setPendingRequests] = useState({});

  // Temel fetch fonksiyonu
  const fetchWithAuth = async (endpoint, options = {}) => {
    try {
      const cacheKey = `${endpoint}-${JSON.stringify(options)}`;
      
      // Eğer aynı istek zaten beklemedeyse, mevcut promise'i döndür
      if (pendingRequests[cacheKey]) {
        console.log(`Pending request reused for: ${endpoint}`);
        return pendingRequests[cacheKey];
      }
      
      // GET istekleri için cache kontrolü
      if (!options.method || options.method === 'GET') {
        if (cache[cacheKey]) {
          console.log(`Cache hit for: ${endpoint}`);
          return cache[cacheKey];
        }
      }

      // Eğer kimlik doğrulaması yapılmışsa, token ekle
      if (isAuthenticated) {
        try {
          const token = await getToken();
          
          // Headers'ı ayarla
          options.headers = {
            ...options.headers,
            Authorization: `Bearer ${token}`
          };
        } catch (error) {
          console.warn('Token alınamadı, istek token olmadan gönderilecek:', error);
        }
      }

      // Varsayılan ayarları ekle
      options = {
        ...options,
        credentials: 'include', // Cookie'leri dahil et
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      };

      console.log(`API isteği yapılıyor: ${endpoint}`, options);

      // İstek promise'ini oluştur ve bekleyen istekler listesine ekle
      const requestPromise = new Promise(async (resolve, reject) => {
        try {
          // İsteği yap
          const response = await fetch(`${API_URL}${endpoint}`, options);
          
          // JSON olmayan yanıtlar için (örn. dosya indirme)
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            
            // Başarısız yanıtlar için hata fırlat
            if (!response.ok) {
              console.error(`API hatası (${response.status}): ${data.error || data.message || 'Bilinmeyen hata'}`);
              throw new Error(data.error || data.message || `HTTP ${response.status}: Bir hata oluştu`);
            }
            
            // GET istekleri için sonucu cache'e ekle
            if (!options.method || options.method === 'GET') {
              setCache(prevCache => ({
                ...prevCache,
                [cacheKey]: data
              }));
            }
            
            resolve(data);
          } else {
            // JSON olmayan yanıtlar için (örn. dosya indirme)
            if (!response.ok) {
              console.error(`API hatası (${response.status}): İçerik türü - ${contentType}`);
              throw new Error(`HTTP ${response.status}: Bir hata oluştu`);
            }
            
            resolve(response);
          }
        } catch (error) {
          console.error(`API error (${endpoint}):`, error);
          reject(error);
        } finally {
          // İstek tamamlandığında bekleyen istekler listesinden çıkar
          setPendingRequests(prev => {
            const newPending = { ...prev };
            delete newPending[cacheKey];
            return newPending;
          });
        }
      });
      
      // İsteği bekleyen istekler listesine ekle
      setPendingRequests(prev => ({
        ...prev,
        [cacheKey]: requestPromise
      }));
      
      return requestPromise;
    } catch (error) {
      console.error(`API error (${endpoint}):`, error);
      throw error;
    }
  };

  // Cache'i temizle
  const clearCache = useCallback((endpoint = null) => {
    if (endpoint) {
      setCache(prevCache => {
        const newCache = { ...prevCache };
        Object.keys(newCache).forEach(key => {
          if (key.startsWith(`${endpoint}-`)) {
            delete newCache[key];
          }
        });
        return newCache;
      });
    } else {
      setCache({});
    }
  }, []);

  // API metotları
  return {
    // Kullanıcı profili
    getUserProfile: () => fetchWithAuth('/user/profile'),
    updateUserProfile: (data) => {
      // Profil güncellendiğinde cache'i temizle
      clearCache('/user/profile');
      return fetchWithAuth('/user/profile', {
        method: 'PUT',
        body: JSON.stringify(data)
      });
    },
    
    // Analiz işlemleri
    getAnalyses: () => fetchWithAuth('/analysis'),
    getAnalysisById: (id) => fetchWithAuth(`/analysis/${id}`),
    createAnalysis: (data) => fetchWithAuth('/analysis', {
      method: 'POST',
      body: JSON.stringify(data)
    }),
    updateAnalysis: (id, data) => fetchWithAuth(`/analysis/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    }),
    deleteAnalysis: (id) => fetchWithAuth(`/analysis/${id}`, {
      method: 'DELETE'
    }),
    
    // Earth Engine işlemleri
    getMapData: (params) => fetchWithAuth('/earth-engine/map-data', {
      method: 'POST',
      body: JSON.stringify(params)
    }),
    
    // AI analiz işlemleri
    analyzeDamage: (data) => fetchWithAuth('/ai-analysis/damage-analysis', {
      method: 'POST',
      body: JSON.stringify(data)
    }),
    checkAIHealth: () => fetchWithAuth('/ai-analysis/health'),
    
    // Test endpoint'i
    testApi: () => fetchWithAuth('/test'),
    testAuth: () => fetchWithAuth('/auth-test'),
    
    // Cache yönetimi
    clearCache
  };
};
