import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fake Stars Detector",
  description: "Detect potentially fake GitHub stars using anomaly detection and account verification",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
