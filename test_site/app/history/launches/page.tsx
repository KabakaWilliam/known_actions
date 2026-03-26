import Link from 'next/link';
import { PRODUCT_LAUNCHES } from '@/lib/content';

export const metadata = { title: 'Product Launches — Northstar Devices' };

export default function LaunchesPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">Home</Link>
        <span className="mx-2">›</span>
        <Link href="/history" className="hover:text-zinc-600">History</Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">Launches</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Product Launches
      </h1>
      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        Every product Northstar Devices has released, with launch year and context.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {PRODUCT_LAUNCHES.map((launch) => (
          <div
            key={`${launch.name}-${launch.year}`}
            className="rounded-xl border border-zinc-200 bg-white p-5"
          >
            <div className="flex items-center justify-between mb-2">
              <h2 className="font-semibold text-zinc-900 text-[15px]">{launch.name}</h2>
              <span className="text-xs font-semibold text-blue-600 tabular-nums bg-blue-50 px-2 py-0.5 rounded">
                {launch.year}
              </span>
            </div>
            <p className="text-sm text-zinc-500 leading-relaxed">{launch.description}</p>
          </div>
        ))}
      </div>

      <div className="mt-10 text-sm text-zinc-500">
        View company milestones →{' '}
        <Link href="/history/timeline" className="text-blue-600 hover:underline">
          Timeline
        </Link>
      </div>
    </div>
  );
}
