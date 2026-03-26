import type { Metadata } from "next";
import { cookies } from "next/headers";
import "./globals.css";
import Nav from "@/test_site/components/Nav";
import Footer from "@/test_site/components/Footer";
import EventLogger from "@/test_site/components/EventLogger";
import { verifySessionCookie, SESSION_COOKIE } from "@/lib/auth";

export const metadata: Metadata = {
  title: "Northstar Devices",
  description: "Precision instruments for professional and everyday use.",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Resolve session_id server-side so EventLogger knows whether to activate
  const cookieStore = await cookies();
  const token = cookieStore.get(SESSION_COOKIE)?.value ?? null;
  let sessionId: string | null = null;
  if (token) {
    const claims = await verifySessionCookie(token);
    sessionId = claims?.session_id ?? null;
  }

  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col bg-zinc-50 text-zinc-900 antialiased">
        <Nav />
        <main className="flex-1">{children}</main>
        <Footer />
        <EventLogger sessionId={sessionId} />
      </body>
    </html>
  );
}
