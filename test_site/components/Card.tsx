'use client';

import Link from 'next/link';
import { logClick } from '@/lib/logger';

interface CardProps {
  title: string;
  description: string;
  href: string;
  meta?: string; // e.g. a badge or short note
}

export default function Card({ title, description, href, meta }: CardProps) {
  return (
    <Link
      href={href}
      onClick={() => logClick(`card-${title.toLowerCase().replace(/\s+/g, '-')}`)}
      className="group block rounded-xl border border-zinc-200 bg-white p-5 hover:border-zinc-300 hover:shadow-sm transition-all"
    >
      {meta && (
        <span className="inline-block mb-2 text-xs font-medium text-zinc-500 bg-zinc-100 px-2 py-0.5 rounded">
          {meta}
        </span>
      )}
      <h3 className="font-semibold text-zinc-900 group-hover:text-blue-600 transition-colors text-[15px]">
        {title}
      </h3>
      <p className="mt-1 text-sm text-zinc-500 leading-relaxed">{description}</p>
    </Link>
  );
}
