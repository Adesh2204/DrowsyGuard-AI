/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,jsx}",
    "./components/**/*.{js,jsx}",
    "./services/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ["Sora", "sans-serif"],
        body: ["Space Grotesk", "sans-serif"]
      },
      boxShadow: {
        neon: "0 0 30px rgba(43, 237, 214, 0.25)",
        panel: "0 18px 50px rgba(6, 20, 35, 0.6)"
      }
    }
  },
  plugins: []
};
