import Card from "@/test_site/components/Card";

const SECTIONS = [
  {
    title: "Product Facts",
    description: "Current devices, specifications, and pricing information.",
    href: "/facts",
    meta: "Devices & Pricing",
  },
  {
    title: "Company History",
    description: "Our timeline, milestones, and product launch history.",
    href: "/history",
    meta: "Timeline & Launches",
  },
  {
    title: "Support",
    description:
      "Shipping policies, warranty information, and customer guidance.",
    href: "/support",
    meta: "Shipping & Warranty",
  },
  {
    title: "Archive",
    description: "Past product lines, retired devices, and historical pricing.",
    href: "/archive",
    meta: "Past Products",
  },
];

export default function HomePage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-16">
      {/* Hero */}
      <div className="mb-14">
        <p className="text-sm font-medium text-blue-600 mb-3 tracking-wide uppercase">
          Northstar Devices
        </p>
        <h1 className="text-4xl font-bold text-zinc-900 tracking-tight leading-tight mb-4">
          Precision instruments.
          <br />
          Built to last.
        </h1>
        <p className="text-lg text-zinc-500 max-w-xl leading-relaxed">
          We design and manufacture compact measurement and monitoring devices
          for professional and everyday use — made in Edinburgh since 2015.
        </p>
      </div>

      {/* Section cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {SECTIONS.map((s) => (
          <Card key={s.href} {...s} />
        ))}
      </div>

      {/* Short about strip */}
      <div className="mt-16 border-t border-zinc-200 pt-10">
        <h2 className="text-base font-semibold text-zinc-700 mb-2">
          About Northstar
        </h2>
        <p className="text-sm text-zinc-500 leading-relaxed max-w-2xl">
          Founded in 2015, Northstar Devices has grown from a two-person
          workshop into a trusted name in portable instrumentation. Our devices
          are used by field engineers, researchers, and enthusiasts across more
          than 30 countries. We expanded our accessories range in 2021 and
          continue to refine each product line annually.
        </p>
      </div>
    </div>
  );
}
