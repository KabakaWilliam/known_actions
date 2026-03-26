import Card from "@/test_site/components/Card";
import Link from "next/link";

export const metadata = { title: "Support — Northstar Devices" };

export default function SupportPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">
          Home
        </Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">Support</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Support
      </h1>
      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        Everything you need to know about ordering, delivery, returns, and
        warranty coverage for Northstar Devices products.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Card
          title="Shipping"
          description="Delivery timelines, international shipping availability, and returns information."
          href="/support/shipping"
          meta="Delivery & Returns"
        />
        <Card
          title="Warranty"
          description="What's covered under the standard Northstar warranty and how to make a claim."
          href="/support/warranty"
          meta="Coverage & Claims"
        />
      </div>
    </div>
  );
}
