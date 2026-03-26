import Link from 'next/link';
import { HISTORY_EVENTS } from '@/lib/content';

export const metadata = { title: 'Timeline — Northstar Devices' };

export default function TimelinePage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">Home</Link>
        <span className="mx-2">›</span>
        <Link href="/history" className="hover:text-zinc-600">History</Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">Timeline</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Company Timeline
      </h1>
      <p className="text-zinc-500 mb-12 max-w-xl leading-relaxed">
        A chronological record of the milestones that have shaped Northstar Devices.
      </p>

      <ol className="relative border-l border-zinc-200 space-y-8 ml-3">
        {HISTORY_EVENTS.map((event) => (
          <li key={event.year} className="pl-8">
            <div className="absolute -left-[9px] w-4 h-4 rounded-full bg-blue-500 border-2 border-white" />
            <div className="flex items-baseline gap-3 mb-1">
              <span className="text-xs font-semibold text-blue-600 tabular-nums">
                {event.year}
              </span>
              <h3 className="font-semibold text-zinc-900 text-[15px]">{event.label}</h3>
            </div>
            <p className="text-sm text-zinc-500 leading-relaxed">{event.detail}</p>
          </li>
        ))}
      </ol>

      <div className="mt-12 text-sm text-zinc-500">
        See all product launches →{' '}
        <Link href="/history/launches" className="text-blue-600 hover:underline">
          Product Launches
        </Link>
      </div>
    </div>
  );
}
