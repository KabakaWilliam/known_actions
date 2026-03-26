import Card from "@/test_site/components/Card";
import Link from "next/link";

export const metadata = { title: "Archive — Northstar Devices" };

export default function ArchivePage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">
          Home
        </Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">Archive</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Archive
      </h1>
      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        Historical records of past product lines, archived pricing, and devices
        that have since been retired.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Card
          title="2023 Product Line"
          description="Pricing and details for the 2023 Northstar Devices line, now superseded by the current range."
          href="/archive/2023-line"
          meta="Archived 2023"
        />
        <Card
          title="Retired Devices"
          description="Products that have been discontinued and are no longer available for purchase."
          href="/archive/retired-devices"
          meta="Discontinued"
        />
      </div>
    </div>
  );
}
