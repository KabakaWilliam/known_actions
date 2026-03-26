import Card from "@/test_site/components/Card";
import Link from "next/link";

export const metadata = { title: "Product Facts — Northstar Devices" };

export default function FactsPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">
          Home
        </Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">Facts</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Product Facts
      </h1>
      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        Current product information for the Northstar Devices range —
        specifications, device details, and up-to-date pricing.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Card
          title="Devices"
          description="Browse the current Northstar product line with full specifications and availability."
          href="/facts/devices"
          meta="Current range"
        />
        <Card
          title="Pricing"
          description="Current retail prices for all active Northstar products, including stock availability."
          href="/facts/pricing"
          meta="Retail prices"
        />
      </div>
    </div>
  );
}
