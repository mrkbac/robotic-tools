import { createRootRoute, Link, Outlet } from "@tanstack/react-router";
import {
  Container,
  Stack,
  Group,
  Title,
  ActionIcon,
  useMantineColorScheme,
  useComputedColorScheme,
} from "@mantine/core";
import { NavigationProgress } from "@mantine/nprogress";
import { IconSun, IconMoon } from "@tabler/icons-react";
import {
  FileProcessorProvider,
  useFileProcessorContext,
} from "../contexts/FileProcessorContext.tsx";
import { GlobalDropZone } from "../components/GlobalDropZone.tsx";

function ColorSchemeToggle() {
  const { setColorScheme } = useMantineColorScheme();
  const computed = useComputedColorScheme("light");
  return (
    <ActionIcon
      variant="default"
      size="lg"
      onClick={() => setColorScheme(computed === "light" ? "dark" : "light")}
      aria-label="Toggle color scheme"
    >
      {computed === "light" ? <IconMoon size={18} /> : <IconSun size={18} />}
    </ActionIcon>
  );
}

function RootLayoutInner() {
  const { handleFilesSelect } = useFileProcessorContext();

  return (
    <>
      <GlobalDropZone onFilesSelect={handleFilesSelect} />
      <NavigationProgress />
      <Container size="xl" py="xl">
        <Stack gap="lg">
          <Group justify="space-between" align="flex-end">
            <Title
              order={2}
              renderRoot={(props) => (
                <Link
                  to="/"
                  {...props}
                  style={{
                    ...props.style,
                    textDecoration: "none",
                    color: "inherit",
                  }}
                />
              )}
            >
              MCAP Web Inspector
            </Title>
            <ColorSchemeToggle />
          </Group>
          <Outlet />
        </Stack>
      </Container>
    </>
  );
}

function RootLayout() {
  return (
    <FileProcessorProvider>
      <RootLayoutInner />
    </FileProcessorProvider>
  );
}

export const Route = createRootRoute({
  component: RootLayout,
});
