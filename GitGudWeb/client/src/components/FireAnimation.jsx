import React, { useEffect, useState } from 'react';
import './FireAnimation.css';

const FireAnimation = () => {
  const [flames, setFlames] = useState([]);

  useEffect(() => {
    // Ekran geniu015fliu011fine gu00f6re alev sayu0131su0131nu0131 belirle
    const flameCount = Math.floor(window.innerWidth / 200) + 2;
    
    // Alevleri oluu015ftur
    const newFlames = [];
    for (let i = 0; i < flameCount; i++) {
      // Her alev iu00e7in rasgele boyut ve pozisyon
      const size = 120 + Math.random() * 80;
      newFlames.push({
        id: i,
        left: `${(i * 100) / flameCount + Math.random() * 10}%`,
        size: size,
        animationDuration: `${1.5 + Math.random() * 1}s`,
        animationDelay: `${Math.random() * 0.5}s`,
        scale: 0.8 + Math.random() * 0.4
      });
    }
    setFlames(newFlames);

    // Ekran boyutu deu011fiu015ftiu011finde alevleri yeniden oluu015ftur
    const handleResize = () => {
      const newFlameCount = Math.floor(window.innerWidth / 200) + 2;
      const resizedFlames = [];
      for (let i = 0; i < newFlameCount; i++) {
        const size = 120 + Math.random() * 80;
        resizedFlames.push({
          id: i,
          left: `${(i * 100) / newFlameCount + Math.random() * 10}%`,
          size: size,
          animationDuration: `${1.5 + Math.random() * 1}s`,
          animationDelay: `${Math.random() * 0.5}s`,
          scale: 0.8 + Math.random() * 0.4
        });
      }
      setFlames(resizedFlames);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="fire-container">
      {flames.map((flame) => (
        <div
          key={flame.id}
          className="flame-wrapper"
          style={{
            left: flame.left,
            animationDuration: flame.animationDuration,
            animationDelay: flame.animationDelay,
          }}
        >
          <svg
            className="flame"
            width={flame.size}
            height={flame.size * 1.2}
            viewBox="0 0 100 120"
            style={{
              transform: `scale(${flame.scale})`,
              filter: `blur(${2 + Math.random() * 3}px)`
            }}
          >
            {/* Du0131u015f alev - turuncu/ku0131rmu0131zu0131 */}
            <path
              className="flame-outer"
              d="M30,110 Q-10,85 15,40 Q25,10 50,5 Q75,10 85,40 Q110,85 70,110 Q50,120 30,110"
            />
            
            {/* Orta alev - turuncu */}
            <path
              className="flame-middle"
              d="M35,105 Q5,80 25,45 Q35,20 50,15 Q65,20 75,45 Q95,80 65,105 Q50,115 35,105"
            />
            
            {/* u0130u00e7 alev - saru0131 */}
            <path
              className="flame-inner"
              d="M40,100 Q20,75 35,50 Q42,30 50,25 Q58,30 65,50 Q80,75 60,100 Q50,110 40,100"
            />
            
            {/* Merkez - beyaz/saru0131 */}
            <path
              className="flame-core"
              d="M45,95 Q35,75 42,60 Q45,50 50,45 Q55,50 58,60 Q65,75 55,95 Q50,100 45,95"
            />
          </svg>
        </div>
      ))}
    </div>
  );
};

export default FireAnimation;
