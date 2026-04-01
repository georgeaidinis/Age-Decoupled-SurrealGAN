import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./styles.css";

class AppErrorBoundary extends React.Component<React.PropsWithChildren, { hasError: boolean; message: string }> {
  constructor(props: React.PropsWithChildren) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, message: error.message };
  }

  override componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("App render failure", error, errorInfo);
  }

  override render() {
    if (this.state.hasError) {
      return (
        <main className="app-shell fallback-shell">
          <section className="panel status-panel">
            <div className="panel-header">
              <h2>Frontend error</h2>
              <p>{this.state.message || "The interface hit a runtime error."}</p>
            </div>
          </section>
        </main>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </React.StrictMode>,
);
