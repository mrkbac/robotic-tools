import { Modal, UnstyledButton } from "@mantine/core";
import { Sparkline } from "@mantine/charts";
import { useDisclosure } from "@mantine/hooks";

interface SparklineModalProps {
  data: number[];
  title: string;
  children: React.ReactNode;
  w?: number;
  h?: number;
}

export function SparklineModal({
  data,
  title,
  children,
  w = 120,
  h = 30,
}: SparklineModalProps) {
  const [opened, { open, close }] = useDisclosure(false);

  return (
    <>
      <UnstyledButton onClick={open} style={{ cursor: "pointer" }}>
        <Sparkline
          w={w}
          h={h}
          data={data}
          curveType="monotone"
          color="blue"
          fillOpacity={0.2}
          strokeWidth={1.5}
        />
      </UnstyledButton>
      <Modal opened={opened} onClose={close} title={title} size="xl">
        {children}
      </Modal>
    </>
  );
}
