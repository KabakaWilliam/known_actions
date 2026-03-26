import Link from 'next/link';
import { RETIRED_DEVICES } from '@/lib/content';

export const metadata = { title: 'Retired Devices — Northstar Devices Archive' };

export default function RetiredDevicesPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="mb-2 text-sm text-zinc-400">
        <Link href="/" className="hover:text-zinc-600">Home</Link>
        <span className="mx-2">›</span>
        <Link href="/archive" className="hover:text-zinc-600">Archive</Link>
        <span className="mx-2">›</span>
        <span className="text-zinc-600">Retired Devices</span>
      </div>

      <h1 className="text-3xl font-bold text-zinc-900 tracking-tight mt-4 mb-3">
        Retired Devices
      </h1>
      <p className="text-zinc-500 mb-10 max-w-xl leading-relaxed">
        Products that have been permanently discontinued. These devices are no longer
        available for purchase, though some spare parts may still be available.
      </p>

      <div className="space-y-4">
        {RETIRED_DEVICES.map((device) => (
          <div
            key={device.name}
            className="rounded-xl border border-zinc-200 bg-white p-6"
          >
            <div className="flex items-start justify-between gap-4 mb-3">
              <div>
                <h2 className="font-semibold text-zinc-900 text-[15px]">{device.name}</h2>
                <p className="text-xs text-zinc-400 mt-0.5">
                  Launched {device.launched} — Retired {device.retired_year}
                </p>
              </div>
              <span className="shrink-0 text-xs font-medium bg-red-50 text-red-600 border border-red-200 px-2 py-0.5 rounded">
                Discontinued
              </span>
            </div>
            <p className="text-sm text-zinc-500 leading-relaxed">{device.reason}</p>
          </div>
        ))}
      </div>

      <p className="mt-8 text-sm text-zinc-500">
        For current products, see{' '}
        <Link href="/facts/devices" className="text-blue-600 hover:underline">
          current devices
        </Link>
        . For 2023 archived pricing, see{' '}
        <Link href="/archive/2023-line" className="text-blue-600 hover:underline">
          the 2023 line
        </Link>
        .
      </p>
    </div>
  );
}
