import { createFileRoute } from "@tanstack/react-router";
import { ForgeApp } from "@/components/forge-app";

export const Route = createFileRoute("/")({ component: ForgeApp });
