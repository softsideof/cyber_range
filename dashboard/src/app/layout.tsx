import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'CyberRange — Autonomous SOC Defense & Adversary Simulation',
  description:
    'Interactive 3D enterprise network attack simulation platform. Autonomous SOC policy defending against APTs, ransomware, and supply chain attacks.',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
