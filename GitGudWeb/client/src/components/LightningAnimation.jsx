import React, { useEffect, useState, useRef } from 'react';
import './LightningAnimation.css';

const LightningAnimation = () => {
  const [lightnings, setLightnings] = useState([]);
  const [flash, setFlash] = useState(false);
  const timeoutRef = useRef(null);
  const intervalRef = useRef(null);

  // Rasgele yu0131ldu0131ru0131m oluu015fturma fonksiyonu
  const createLightning = () => {
    const newLightnings = [];
    // 3-7 arasu0131 rasgele yu0131ldu0131ru0131m oluu015ftur - sayu0131 artu0131ru0131ldu0131
    const count = Math.floor(Math.random() * 5) + 3;
    
    for (let i = 0; i < count; i++) {
      // Rasgele pozisyon ve boyut
      const left = `${Math.random() * 90 + 5}%`;
      const height = `${Math.random() * 40 + 60}%`;
      const width = `${Math.random() * 5 + 2}px`;
      const delay = `${Math.random() * 0.2}s`;
      const duration = `${Math.random() * 0.3 + 0.7}s`;
      
      // Ana yu0131ldu0131ru0131m
      newLightnings.push({
        id: `lightning-${Date.now()}-${i}`,
        style: {
          left,
          height,
          width,
          animationDelay: delay,
          animationDuration: duration,
          animationName: 'lightning-strike',
          animationTimingFunction: 'ease-in-out',
          animationIterationCount: 1
        }
      });
      
      // Yu0131ldu0131ru0131m dallanmalaru0131 (rasgele) - daha fazla dallanma
      if (Math.random() > 0.3) {
        const branchCount = Math.floor(Math.random() * 5) + 2;
        for (let j = 0; j < branchCount; j++) {
          const branchTop = `${Math.random() * 70 + 10}%`;
          const branchWidth = `${Math.random() * 2 + 1}px`;
          const branchHeight = `${Math.random() * 20 + 10}%`;
          const branchAngle = Math.random() > 0.5 ? 
            `${Math.random() * 30 + 10}deg` : 
            `-${Math.random() * 30 + 10}deg`;
          
          newLightnings.push({
            id: `branch-${Date.now()}-${i}-${j}`,
            style: {
              left,
              top: branchTop,
              width: branchWidth,
              height: branchHeight,
              transform: `scaleY(0) rotate(${branchAngle})`,
              transformOrigin: 'top',
              animationDelay: `calc(${delay} + 0.1s)`,
              animationDuration: `${Math.random() * 0.2 + 0.3}s`,
              animationName: 'lightning-strike',
              animationTimingFunction: 'ease-in-out',
              animationIterationCount: 1
            }
          });
        }
      }
    }
    
    // Yu0131ldu0131ru0131m u015fimu015feu011fi efekti
    setFlash(true);
    setTimeout(() => setFlash(false), 500);
    
    return newLightnings;
  };

  useEffect(() => {
    // Rasgele aralu0131klarla yu0131ldu0131ru0131m oluu015ftur
    const triggerLightning = () => {
      setLightnings(createLightning());
      
      // Yu0131ldu0131ru0131mlaru0131 temizle
      timeoutRef.current = setTimeout(() => {
        setLightnings([]);
      }, 2000);
      
      // Sonraki yu0131ldu0131ru0131m iu00e7in rasgele su00fcre belirle (2-6 saniye) - daha su0131k yu0131ldu0131ru0131m
      const nextInterval = Math.random() * 4000 + 2000;
      intervalRef.current = setTimeout(triggerLightning, nextInterval);
    };
    
    // u0130lk yu0131ldu0131ru0131m iu00e7in 1-3 saniye bekle - daha hemen bau015fla
    const initialDelay = Math.random() * 2000 + 1000;
    intervalRef.current = setTimeout(triggerLightning, initialDelay);
    
    return () => {
      clearTimeout(timeoutRef.current);
      clearTimeout(intervalRef.current);
    };
  }, []);

  return (
    <div className="lightning-container">
      {lightnings.map((lightning) => (
        <div
          key={lightning.id}
          className="lightning"
          style={lightning.style}
        />
      ))}
      <div 
        className="lightning-flash" 
        style={{
          animation: flash ? 'lightning-flash 1s ease-out' : 'none'
        }}
      />
    </div>
  );
};

export default LightningAnimation;
