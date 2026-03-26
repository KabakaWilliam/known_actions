import Link from "next/link";
import { WARRANTY_POLICY } from "@/lib/content";
import Accordion from "@/test_site/components/Accordion";

export const metadata = { title: "Warranty — Northstar Devices" };

export default function WarrantyPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">
          Home
        </Link>
        <span className="mx-2">›</span>
        <Link href="/support" className="hover:text-zinc-600">
          Support
        </Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">Warranty</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Warranty Information
      </h1>
      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        All Northstar Devices products come with a standard{" "}
        {WARRANTY_POLICY.duration_years}-year limited warranty from date of
        purchase. Read below to understand what is and isn't covered.
      </p>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-10">
        <div className="rounded-xl border border-zinc-200 bg-white p-5">
          <p className="text-xs font-medium text-zinc-400 uppercase tracking-wide mb-1">
            Duration
          </p>
          <p className="text-2xl font-bold text-zinc-900">
            {WARRANTY_POLICY.duration_years} years
          </p>
          <p className="text-sm text-zinc-500 mt-1">from date of purchase</p>
        </div>
        <div className="rounded-xl border border-zinc-200 bg-white p-5">
          <p className="text-xs font-medium text-zinc-400 uppercase tracking-wide mb-1">
            Coverage
          </p>
          <div className="space-y-1 mt-1">
            <p className="text-sm text-zinc-700">
              <span className="text-green-600 font-semibold">✓</span>{" "}
              Manufacturing defects
            </p>
            <p className="text-sm text-zinc-700">
              <span className="text-green-600 font-semibold">✓</span> Component
              failures
            </p>
            <p className="text-sm text-zinc-500">
              <span className="text-red-500 font-semibold">✗</span> Accidental
              damage
            </p>
            <p className="text-sm text-zinc-500">
              <span className="text-red-500 font-semibold">✗</span> Wear and
              tear
            </p>
          </div>
        </div>
      </div>

      {/* FAQ Accordion */}
      <h2 className="text-lg font-semibold text-zinc-800 mb-4">
        Frequently Asked Questions
      </h2>
      <Accordion items={WARRANTY_POLICY.faqs} />

      <p className="mt-8 text-sm text-zinc-500">
        To initiate a warranty claim, contact our support team with your proof
        of purchase. For shipping queries, visit{" "}
        <Link
          href="/support/shipping"
          className="text-blue-600 hover:underline"
        >
          Shipping
        </Link>
        .
      </p>
    </div>
  );
}
