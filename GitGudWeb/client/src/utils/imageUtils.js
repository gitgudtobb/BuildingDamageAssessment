/**
 * Görüntü işleme yardımcı fonksiyonları
 */

/**
 * Base64 formatındaki veriyi düzeltir
 * @param {string} base64String - Düzeltilecek base64 string
 * @returns {string} - Düzeltilmiş base64 string
 */
export const fixBase64Padding = (base64String) => {
  // Base64 prefix'i kaldır
  let cleanedString = base64String;
  if (cleanedString.includes(',')) {
    cleanedString = cleanedString.split(',')[1];
  }
  
  // Base64 padding düzeltme
  while (cleanedString.length % 4 !== 0) {
    cleanedString += '=';
  }
  
  // Prefix'i tekrar ekle
  if (base64String.includes(',')) {
    return base64String.split(',')[0] + ',' + cleanedString;
  }
  
  return cleanedString;
};

/**
 * Bir görüntüyü belirli bir boyuta yeniden boyutlandırır
 * @param {string} base64Image - Base64 formatındaki görüntü
 * @param {number} maxWidth - Maksimum genişlik
 * @param {number} maxHeight - Maksimum yükseklik
 * @returns {Promise<string>} - Yeniden boyutlandırılmış görüntü (base64)
 */
export const resizeImage = (base64Image, maxWidth = 800, maxHeight = 600) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      let width = img.width;
      let height = img.height;
      
      // En-boy oranını koru
      if (width > maxWidth) {
        height = Math.round(height * (maxWidth / width));
        width = maxWidth;
      }
      
      if (height > maxHeight) {
        width = Math.round(width * (maxHeight / height));
        height = maxHeight;
      }
      
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, width, height);
      
      // PNG formatında daha iyi sonuç verebilir
      const resizedImage = canvas.toDataURL('image/png');
      resolve(resizedImage);
    };
    
    img.onerror = (error) => {
      reject(error);
    };
    
    img.src = base64Image;
  });
};
