import React, { useEffect, useState } from 'react';
import './SmokeAnimation.css';

const SmokeAnimation = () => {
  const [smokeElements, setSmokeElements] = useState([]);

  useEffect(() => {
    // Bau015flangiu00e7ta duman elementlerini oluu015ftur
    createSmokeElements();
    
    // Belirli aralu0131klarla yeni duman elementleri oluu015ftur
    const interval = setInterval(() => {
      createSmokeElements(3); // Her seferinde 3 yeni duman elementi
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Duman elementleri oluu015fturma fonksiyonu
  const createSmokeElements = (count = 10) => {
    const newSmokeElements = [];
    
    for (let i = 0; i < count; i++) {
      newSmokeElements.push({
        id: `smoke-${Date.now()}-${i}`,
        left: `${Math.random() * 100}%`,
        size: `${100 + Math.random() * 200}px`,
        delay: `${Math.random() * 3}s`,
        duration: `${5 + Math.random() * 5}s`,
        opacity: 0.1 + Math.random() * 0.3
      });
    }
    
    setSmokeElements(prev => {
      // En fazla 30 duman elementi olsun, eskilerini kaldur0131r
      const updated = [...prev, ...newSmokeElements];
      if (updated.length > 30) {
        return updated.slice(updated.length - 30);
      }
      return updated;
    });
  };

  return (
    <div className="smoke-container">
      {smokeElements.map((smoke) => (
        <div
          key={smoke.id}
          className="smoke"
          style={{
            left: smoke.left,
            width: smoke.size,
            height: smoke.size,
            animationDelay: smoke.delay,
            animationDuration: smoke.duration,
            opacity: smoke.opacity,
          }}
        />
      ))}
    </div>
  );
};

export default SmokeAnimation;
