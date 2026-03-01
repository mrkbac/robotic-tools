import { createRootRoute, Outlet } from "@tanstack/react-router";
import {
  Container,
  Stack,
  Group,
  Title,
  ActionIcon,
  Progress,
  useMantineColorScheme,
  useComputedColorScheme,
} from "@mantine/core";
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
  const { loading, progress, handleFilesSelect } = useFileProcessorContext();

  return (
    <>
      <GlobalDropZone onFilesSelect={handleFilesSelect} />
      {loading && (
        <Progress
          value={progress}
          size="xs"
          style={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 999 }}
        />
      )}
      <Container size="xl" py="xl">
        <Stack gap="lg">
          <Group justify="space-between" align="flex-end">
            <Title order={2}>MCAP Web Inspector</Title>
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
