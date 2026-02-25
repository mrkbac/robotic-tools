import { createRootRoute, Outlet } from "@tanstack/react-router";
import {
  Container,
  Stack,
  Group,
  Title,
  ActionIcon,
  useMantineColorScheme,
  useComputedColorScheme,
} from "@mantine/core";
import { IconSun, IconMoon } from "@tabler/icons-react";

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

function RootLayout() {
  return (
    <Container size="xl" py="xl">
      <Stack gap="lg">
        <Group justify="space-between" align="flex-end">
          <Title order={2}>MCAP Web Inspector</Title>
          <ColorSchemeToggle />
        </Group>
        <Outlet />
      </Stack>
    </Container>
  );
}

export const Route = createRootRoute({
  component: RootLayout,
});
