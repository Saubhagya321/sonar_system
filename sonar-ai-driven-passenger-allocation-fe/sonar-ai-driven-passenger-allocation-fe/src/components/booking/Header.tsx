import React from 'react';
import SonarLogo from '@/assets/sonar-logo.svg';

interface HeaderProps {
  className?: string;
}

export const Header: React.FC<HeaderProps> = ({ className = '' }) => {
  return (
    <header className={`fixed top-0 left-0 right-0 h-16 bg-[#00207D] z-50 ${className}`}>
      <div className="h-full flex items-center px-10">
        <div className="flex flex-col">
          <img
          src={SonarLogo}
          alt="Sonar Logo"
          className="h-8 w-auto object-contain"
        />
        </div>
      </div>
    </header>
  );
};
