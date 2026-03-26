'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { logClick } from '@/lib/logger';

const NAV_LINKS = [
  { href: '/', label: 'Home' },
  { href: '/facts', label: 'Facts' },
  { href: '/history', label: 'History' },
  { href: '/support', label: 'Support' },
  { href: '/archive', label: 'Archive' },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <header className="border-b border-zinc-200 bg-white">
      <div className="mx-auto max-w-5xl px-6 flex items-center justify-between h-14">
        <Link
          href="/"
          className="font-semibold text-zinc-900 tracking-tight text-[15px]"
          onClick={() => logClick('nav-logo')}
        >
          Northstar Devices
        </Link>
        <nav className="flex items-center gap-1">
          {NAV_LINKS.map(({ href, label }) => {
            const active =
              href === '/' ? pathname === '/' : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                onClick={() => logClick(`nav-${label.toLowerCase()}`)}
                className={`px-3 py-1.5 rounded text-sm transition-colors ${
                  active
                    ? 'bg-zinc-100 text-zinc-900 font-medium'
                    : 'text-zinc-600 hover:text-zinc-900 hover:bg-zinc-50'
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
