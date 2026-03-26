export default function Footer() {
  return (
    <footer className="border-t border-zinc-200 bg-white mt-auto">
      <div className="mx-auto max-w-5xl px-6 py-6 flex items-center justify-between text-xs text-zinc-400">
        <span>© {new Date().getFullYear()} Northstar Devices Ltd. All rights reserved.</span>
        <span>Edinburgh, Scotland</span>
      </div>
    </footer>
  );
}
