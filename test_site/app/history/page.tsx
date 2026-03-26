import Card from "@/test_site/components/Card";
import Link from "next/link";

export const metadata = { title: "History — Northstar Devices" };

export default function HistoryPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">
          Home
        </Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">History</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Our History
      </h1>
      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        From a small workshop in Edinburgh to a product range used worldwide —
        explore our milestones and product launch history.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Card
          title="Timeline"
          description="Key milestones in the Northstar Devices story, from founding through to today."
          href="/history/timeline"
          meta="Company milestones"
        />
        <Card
          title="Product Launches"
          description="A record of every product we have launched, with release dates and context."
          href="/history/launches"
          meta="Product history"
        />
      </div>
    </div>
  );
}
