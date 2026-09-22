import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SVKM NMIMS Voice Assistant',
  description: 'Multilingual voice assistant for SVKM NMIMS Global University, Dhule',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50">{children}</body>
    </html>
  );
}
